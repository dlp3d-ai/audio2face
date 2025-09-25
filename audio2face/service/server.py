import asyncio
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import (
    APIRouter,
    FastAPI,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..apis.builder import build_api
from ..apis.streaming_audio2face_v1 import (
    StreamingAudio2FaceV1,
    StreamingAudio2FaceV1ChunkBody,
    StreamingAudio2FaceV1ChunkEnd,
    StreamingAudio2FaceV1ChunkStart,
)
from ..data_structures.face_clip import FaceClip
from ..utils.super import Super
from .exceptions import NoLogFileException, register_error_handlers

try:
    from ..data_structures import audio2face_v1_pb2 as a2f_v1_pb2
    v1_pb2_imported = True
    v1_pb2_traceback_str = None
except ImportError:
    v1_pb2_imported = False
    v1_pb2_traceback_str = traceback.format_exc()


class Audio2FaceV1ResponseChunkStart(BaseModel):
    """Response model for the first chunk of audio2face streaming response.

    Contains metadata about the blendshape names and data type for the
    streaming audio2face response.
    """
    blendshape_names: list[str]
    dtype: str

class Audio2FaceV1ResponseChunkBody(BaseModel):
    """Response model for body chunks of audio2face streaming response.

    Represents the data chunks containing actual blendshape animation data.
    """
    pass

class Audio2FaceV1ResponseChunkEnd(BaseModel):
    """Response model for the end chunk of audio2face streaming response.

    Signals the end of the streaming audio2face response.
    """
    pass

class FastAPIServer(Super):
    """Backend server for handling HTTP requests and WebSocket connections.

    Provides streaming audio-to-face animation processing services,
    supporting WebSocket real-time communication and HTTP interfaces for
    health checks, log viewing, and other functionalities.
    """

    def __init__(
        self,
        python_api_cfg: dict,
        response_chunk_n_frames: int = 10,
        max_workers: int = 4,
        enable_cors: bool = False,
        host: str = '0.0.0.0',
        port: int = 80,
        startup_event_listener: None | list = None,
        shutdown_event_listener: None | list = None,
        logger_cfg: None | dict = None,
    ) -> None:
        """Initialize the FastAPI server.

        Args:
            python_api_cfg (dict):
                Python API configuration dictionary.
            response_chunk_n_frames (int, optional):
                Number of frames contained in response data chunks.
                Defaults to 10.
            max_workers (int, optional):
                Maximum number of worker threads. Defaults to 4.
            enable_cors (bool, optional):
                Whether to enable CORS cross-origin support.
                Defaults to False.
            host (str, optional):
                Server listening address. Defaults to '0.0.0.0'.
            port (int, optional):
                Server listening port. Defaults to 80.
            startup_event_listener (None | list, optional):
                List of startup event listeners. Defaults to None.
            shutdown_event_listener (None | list, optional):
                List of shutdown event listeners. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.

        Raises:
            ImportError:
                When protobuf module import fails.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.python_api_cfg = python_api_cfg
        self.host = host
        self.port = port
        self.response_chunk_n_frames = response_chunk_n_frames
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.python_api: StreamingAudio2FaceV1 | None = None
        self._build_python_api()
        # for tailing the log file
        log_path = None
        for logger_handler in self.logger.handlers:
            if hasattr(logger_handler, "baseFilename"):
                log_path = logger_handler.baseFilename
                break
        self.log_path = log_path
        self.templates = Jinja2Templates(directory="templates")
        self.app = FastAPI()
        self.enable_cors = enable_cors
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=['*'],
                allow_credentials=True,
                allow_methods=['*'],
                allow_headers=['*'],
            )
        if startup_event_listener is not None:
            for listener in startup_event_listener:
                self.app.add_event_handler('startup', listener)
        if hasattr(self.python_api, 'unitalker'):
            self.app.add_event_handler(
                'startup', self.python_api.unitalker.warmup)
        if hasattr(self.python_api, 'feature_extractor'):
            self.app.add_event_handler(
                'startup', self.python_api.feature_extractor.warmup)
        if shutdown_event_listener is not None:
            for listener in shutdown_event_listener:
                self.app.add_event_handler('shutdown', listener)
        register_error_handlers(self.app)
        self.app.middleware('http')(self._print_request_id)
        self.asyncio_tasks = set()
        if not v1_pb2_imported:
            msg = ('protobuf module import failed, PB-related interfaces '
                   'cannot be used, error details:\n')
            msg += v1_pb2_traceback_str
            self.logger.error(msg)
            raise ImportError(msg)

    def _build_python_api(self):
        """Build Python API instance.

        Asynchronously builds the audio-to-facial-expression API instance
        during server startup, configuring thread pool executor and logger.
        """
        python_api_cfg = self.python_api_cfg.copy()
        python_api_cfg['thread_pool_executor'] = self.executor
        python_api_cfg['logger_cfg'] = self.logger_cfg
        python_api = build_api(python_api_cfg)
        self.python_api = python_api

    async def _print_request_id(self, request: Request, call_next):
        """Print request ID and request information.

        Middleware function for logging detailed information of each HTTP request,
        including request ID, method, path, status code, and processing time.

        Args:
            request (Request):
                FastAPI request object.
            call_next:
                Next processing function.

        Returns:
            Response:
                HTTP response object.
        """
        # print x-request-id for tracking request
        x_request_id = request.headers.get('x-request-id')
        start = time.time()
        response = await call_next(request)
        text = f'{request.method} {request.url.path} {response.status_code}\
            cost: {time.time() - start:.3f}s'

        if x_request_id:
            text = f'x-request-id:{x_request_id} ' + text
        self.logger.debug(text)
        return response

    def _add_api_routes(self, router: APIRouter) -> None:
        """Add API routes.

        Register all HTTP and WebSocket routes, including health checks,
        log viewing, file downloads, and streaming audio processing interfaces.

        Args:
            router (APIRouter):
                FastAPI router instance.
        """
        # GET
        router.add_api_route(
            "/",
            self.root,
            methods=["GET"],
        )
        router.add_api_route(
            '/health',
            endpoint=self.health,
            status_code=200,
            methods=['GET'],
        )
        router.add_api_route(
            "/tail_log/{n_lines}",
            self.tail_log,
            methods=["GET"],
            status_code=200,
            response_model=str,
        )
        router.add_api_route(
            "/dowload_log_file",
            self.dowload_log_file,
            methods=["GET"],
            status_code=200,
        )
        # Websocket
        router.add_api_websocket_route(
            "/api/v1/streaming_audio2face/ws",
            endpoint=self.streaming_audio2face_v1_ws,
        )

    def run(self) -> None:
        """Start the FastAPI server.

        Start the server according to configuration, register all routes
        and begin listening for HTTP requests.
        """
        router = APIRouter()
        self._add_api_routes(router)
        self.app.include_router(router)
        uvicorn.run(self.app, host=self.host, port=self.port)

    def root(self) -> RedirectResponse:
        """Root path redirect to API documentation.

        Returns:
            RedirectResponse:
                Response object redirecting to /docs.
        """
        return RedirectResponse(url="/docs")

    async def dowload_log_file(self) -> Response:
        """Download log file.

        Returns:
            Response:
                Response object containing log file content.

        Raises:
            NoLogFileException:
                When no log file is found.
        """
        if self.log_path is None:
            msg = "No log file found."
            self.logger.error(msg)
            raise NoLogFileException(status_code=503, detail=msg)
        with open(self.log_path, "rb") as f:
            resp = Response(content=f.read(), media_type="application/octet-stream")
            base_name = os.path.basename(self.log_path)
            resp.headers["Content-Disposition"] = f"attachment; filename={base_name}"
            return resp

    async def tail_log(self, request: Request, n_lines: int) -> str:
        """Return the last n lines of the log file.

        Args:
            request (Request):
                FastAPI request object.
            n_lines (int):
                Number of log lines to return.

        Returns:
            str:
                Rendered HTML template containing log content.

        Raises:
            NoLogFileException:
                When no log file is found.
        """
        if self.log_path is None:
            msg = "No log file found."
            self.logger.error(msg)
            raise NoLogFileException(status_code=503, detail=msg)
        # read the last n_lines lines from the log file
        with open(self.log_path, encoding="utf-8") as f:
            lines = f.readlines()
            n_lines = min(int(n_lines), len(lines))
            log_content = "".join(lines[-n_lines:])
        # render template and return
        return self.templates.TemplateResponse(
            "log_template.html", {"request": request, "log_content": log_content})

    async def streaming_audio2face_v1_ws(
        self,
        websocket: WebSocket,
    ):
        """Handle WebSocket connection from client.

        Receives audio data sent by client, performs streaming processing
        and returns facial expression animation data. Supports protobuf format
        data transmission.

        Args:
            websocket (WebSocket):
                WebSocket connection object.

        Note:
            This method handles the complete streaming audio-to-facial-expression
            conversion process, including data reception, processing, and response
            transmission.
        """
        stream_ended = False
        await websocket.accept()
        request_id = 'no_request_id'
        pcm_duration = 0.0
        api_time_cost = 0.0
        try:
            loop = asyncio.get_event_loop()
            pb_bytes = await websocket.receive_bytes()
            pb_request = a2f_v1_pb2.Audio2FaceBlendshapeV1Request()
            await loop.run_in_executor(
                self.executor,
                pb_request.ParseFromString,
                pb_bytes
            )
            if pb_request.class_name != 'StreamingAudio2FaceV1ChunkStart':
                msg = 'Expected StreamingAudio2FaceV1ChunkStart, ' +\
                    f'but received class_name: {pb_request.class_name}'
                self.logger.error(msg)
                await websocket.close(code=1008, reason=msg)
                return
            request_id = pb_request.request_id
            if pb_request.response_chunk_n_frames == 0:
                response_chunk_n_frames = self.response_chunk_n_frames
            else:
                response_chunk_n_frames = pb_request.response_chunk_n_frames
            callback_instance = WebsocketFaceClipCallback(
                request_id=request_id,
                websocket=websocket,
                response_chunk_n_frames=response_chunk_n_frames,
                thread_pool_executor=self.executor,
                logger_cfg=self.logger_cfg
            )
            bytes_per_second = pb_request.sample_rate * \
                pb_request.sample_width * pb_request.n_channels
            item = StreamingAudio2FaceV1ChunkStart(
                request_id=request_id,
                sample_rate=pb_request.sample_rate,
                sample_width=pb_request.sample_width,
                n_channels=pb_request.n_channels,
                callback=callback_instance.send_face_clip,
                profile_name=pb_request.profile_name,
            )
            await self.python_api.handle_chunk_start(item)
            while True:
                pb_bytes = await websocket.receive_bytes()
                pb_request = a2f_v1_pb2.Audio2FaceBlendshapeV1Request()
                await loop.run_in_executor(
                    self.executor,
                    pb_request.ParseFromString,
                    pb_bytes
                )
                if pb_request.class_name == 'StreamingAudio2FaceV1ChunkBody':
                    if len(pb_request.offset_name) == 0:
                        offset_name = None
                    else:
                        offset_name = pb_request.offset_name
                    item = StreamingAudio2FaceV1ChunkBody(
                        request_id=request_id,
                        pcm_bytes=pb_request.pcm_bytes,
                        offset_name=offset_name
                    )
                    start_time = time.time()
                    await self.python_api.handle_chunk_body(item)
                    api_time_cost += time.time() - start_time
                    pcm_duration += len(pb_request.pcm_bytes) / bytes_per_second
                elif pb_request.class_name == 'StreamingAudio2FaceV1ChunkEnd':
                    item = StreamingAudio2FaceV1ChunkEnd(
                        request_id=request_id
                    )
                    start_time = time.time()
                    await self.python_api.handle_chunk_end(item)
                    api_time_cost += time.time() - start_time
                    stream_ended = True
                else:
                    msg = 'Expected StreamingAudio2FaceV1ChunkBody or ' +\
                        'StreamingAudio2FaceV1ChunkEnd, ' +\
                        f'but received class_name: {pb_request.class_name}'
                    self.logger.error(msg)
                    await websocket.close(code=1008, reason=msg)
                    return
        except WebSocketDisconnect:
            msg = f"Connection with request ID {request_id} disconnected by user, " +\
                f"audio duration: {pcm_duration:.3f}s, " +\
                f"A2F API total time cost: {api_time_cost:.3f}s"
            if stream_ended:
                self.logger.info(msg)
            else:
                msg = msg[:-1] + ", but streaming generation did not end normally."
                self.logger.warning(msg)

    async def health(self) -> JSONResponse:
        """Health check endpoint.

        Returns:
            JSONResponse:
                JSON response containing 'OK' status.
        """
        resp = JSONResponse(content='OK')
        return resp

class WebsocketFaceClipCallback(Super):
    """WebSocket facial expression data callback handler.

    Responsible for sending processed facial expression data to clients via
    WebSocket, supporting chunked transmission and protobuf serialization.
    """

    def __init__(
            self,
            request_id: str,
            websocket: WebSocket,
            response_chunk_n_frames: int,
            thread_pool_executor: ThreadPoolExecutor,
            logger_cfg: None | dict = None):
        """Initialize WebSocket callback handler.

        Args:
            request_id (str):
                Request ID for identifying the current processing session.
            websocket (WebSocket):
                WebSocket connection object.
            response_chunk_n_frames (int):
                Number of frames contained in response data chunks.
            thread_pool_executor (ThreadPoolExecutor):
                Thread pool executor for asynchronous serialization.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.request_id = request_id
        self.websocket = websocket
        self.response_chunk_n_frames = response_chunk_n_frames
        self.thread_pool_executor = thread_pool_executor
        self.first_chunk = True

    async def send_face_clip(self, face_clip: FaceClip | None) -> None:
        """Send facial expression data to client.

        Serializes FaceClip data to protobuf format and sends via WebSocket.
        The first data chunk contains metadata information, subsequent chunks
        contain animation data.

        Args:
            face_clip (FaceClip | None):
                Facial expression animation clip to send.
                None indicates end of streaming transmission.
        """
        loop = asyncio.get_event_loop()
        if face_clip is None:
            pb_request = a2f_v1_pb2.Audio2FaceBlendshapeV1Response()
            pb_request.class_name = 'Audio2FaceV1ResponseChunkEnd'
            pb_response_bytes = await loop.run_in_executor(
                self.thread_pool_executor,
                pb_request.SerializeToString
            )
            await self.websocket.send_bytes(pb_response_bytes)
            return
        if self.first_chunk:
            pb_request = a2f_v1_pb2.Audio2FaceBlendshapeV1Response()
            pb_request.class_name = 'Audio2FaceV1ResponseChunkStart'
            pb_request.blendshape_names.extend(face_clip.blendshape_names)
            pb_request.dtype = str(face_clip.blendshape_values.dtype)
            pb_response_bytes = await loop.run_in_executor(
                self.thread_pool_executor,
                pb_request.SerializeToString
            )
            await self.websocket.send_bytes(pb_response_bytes)
            self.first_chunk = False
        for start_idx in range(0, len(face_clip), self.response_chunk_n_frames):
            end_idx = min(start_idx + self.response_chunk_n_frames, len(face_clip))
            bs_value_slice = face_clip.blendshape_values[start_idx:end_idx]
            data = bs_value_slice.tobytes()
            pb_request = a2f_v1_pb2.Audio2FaceBlendshapeV1Response()
            pb_request.class_name = 'Audio2FaceV1ResponseChunkBody'
            pb_request.data = data
            pb_response_bytes = await loop.run_in_executor(
                self.thread_pool_executor,
                pb_request.SerializeToString
            )
            await self.websocket.send_bytes(pb_response_bytes)

