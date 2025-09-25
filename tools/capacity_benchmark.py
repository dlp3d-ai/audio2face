import argparse
import asyncio
import os
import ssl
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import websockets

from audio2face.data_structures import audio2face_v1_pb2 as a2f_v1_pb2
from audio2face.utils.log import setup_logger

CHUNK_DURATION = 0.04

def parse_args():
    """Parse command line arguments for capacity benchmarking.

    Parses command line arguments required for capacity benchmarking,
    including server configuration, input parameters, and benchmark settings.

    Returns:
        argparse.Namespace: Parsed command line arguments object containing:
            - host: Server host address
            - port: Server port number
            - disable_verify: Whether to disable SSL verification
            - input_dir: Input directory containing wav audio files
            - speed_tolerance: Tolerance for A2F service average request speed
            - latency_tolerance: Tolerance for A2F service first response latency
            - users_upperbound: Upper bound for benchmark user count
            - users_lowerbound: Lower bound for benchmark user count
            - user_step: Step size for benchmark user count
            - step_time: Duration for each user count level in benchmark
    """
    parser = argparse.ArgumentParser()
    # Server configuration arguments
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host address.")
    parser.add_argument(
        "--port", type=int, default=8080, help="Server port number.")
    parser.add_argument(
        "--disable_verify", action="store_true",
        help="Whether to disable SSL verification.")
    # Input configuration arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing one or more wav audio files for testing.")
    parser.add_argument(
        "--speed_tolerance",
        type=float,
        default=1.5,
        help="Tolerance for A2F service average request speed.")
    parser.add_argument(
        "--latency_tolerance",
        type=float,
        default=0.3,
        help="Tolerance for A2F service first response latency.")
    # Benchmark configuration arguments
    parser.add_argument(
        "--users_upperbound",
        type=int,
        default=4,
        help="Upper bound for benchmark user count.")
    parser.add_argument("--users_lowerbound",
        type=int,
        default=1,
        help="Lower bound for benchmark user count.")
    parser.add_argument(
        "--user_step",
        type=int,
        default=1,
        help="Step size for benchmark user count.")
    parser.add_argument(
        "--step_time",
        type=float,
        default=30,
        help="Duration for each user count level in benchmark.")
    return parser.parse_args()

async def main():
    """Execute the main capacity benchmarking function.

    Loads audio files and progressively increases concurrent user count to
    benchmark and validate A2F service performance and stability. The test
    verifies that each request meets latency and speed tolerance requirements.

    Execution flow:
    1. Parse command line arguments
    2. Load all wav audio files from the specified directory
    3. Start from the lower bound user count and gradually increase to upper bound
    4. Run tests for the specified duration at each user count level
    5. Validate latency and speed metrics for each request
    6. Log test results and proceed to the next user count configuration
    """
    args = parse_args()
    logger = setup_logger(
        logger_name="capacity_benchmark",
    )
    # Load audio files
    file_names = [f for f in os.listdir(args.input_dir) if f.endswith(".wav")]
    audio_files = list()
    for file_name in file_names:
        file_path = os.path.join(args.input_dir, file_name)
        with wave.open(file_path, "rb") as wav_file:
            parameters = wav_file.getparams()
            audio_bytes = wav_file.readframes(wav_file.getnframes())
            audio_data = dict(
                sample_rate=parameters.framerate,
                channels=parameters.nchannels,
                sample_width=parameters.sampwidth,
                audio_bytes=audio_bytes,
                duration=wav_file.getnframes() / parameters.framerate,
            )
            audio_files.append(audio_data)
    logger.info(f"Loaded {len(audio_files)} audio files.")
    n_users = args.users_lowerbound
    thread_pool_executor = ThreadPoolExecutor(max_workers=args.users_upperbound)
    while n_users <= args.users_upperbound:
        logger.info(
            f"Running benchmark with {n_users} users for {args.step_time} seconds, " +
            f"speed_tolerance={args.speed_tolerance}, " +
            f"latency_tolerance={args.latency_tolerance}...")
        client_pool = [None for _ in range(n_users)]
        step_start_time = time.time()
        audio_idx = 0
        # Create clients and start running
        while time.time() - step_start_time < args.step_time:
            for position, client_tuple in enumerate(client_pool):
                if client_tuple is None:
                    client = BenchmarkClient(
                        host=args.host,
                        port=args.port,
                        disable_verify=args.disable_verify,
                        thread_pool_executor=thread_pool_executor,
                        sample_rate=audio_files[audio_idx]['sample_rate'],
                        channels=audio_files[audio_idx]['channels'],
                        sample_width=audio_files[audio_idx]['sample_width'],
                        audio_bytes=audio_files[audio_idx]['audio_bytes'],
                        audio_duration=audio_files[audio_idx]['duration'],
                        chunk_duration=CHUNK_DURATION,
                        timeout=args.step_time,
                    )
                    task = asyncio.create_task(client.run())
                    client_pool[position] = (client, task)
                    audio_idx = (audio_idx + 1) % len(audio_files)
                else:
                    client, task = client_tuple
                    if client.state == "running":
                        pass
                    elif client.state == "finished":
                        latency = await client.get_latency()
                        if latency > args.latency_tolerance:
                            raise ValueError(
                                f"Latency {latency} is greater than the " +
                                f"tolerance {args.latency_tolerance}")
                        speed = await client.get_speed()
                        if speed < args.speed_tolerance:
                            raise ValueError(
                                f"Speed {speed} is less than the " +
                                f"tolerance {args.speed_tolerance}")
                        client_pool[position] = None
                    else:
                        raise ValueError(
                            f"Client {position} is in an invalid state: {client.state}")
            await asyncio.sleep(0.01)
        # Wait for all clients to finish
        for client_tuple in client_pool:
            if client_tuple is not None:
                client, task = client_tuple
                if client.state == "running":
                    await task
        # Log results
        logger.info(
            f"Benchmark with {n_users} users succeeded, " +
            f"duration={args.step_time}, " +
            f"latency<={args.latency_tolerance}, " +
            f"speed>={args.speed_tolerance}")
        # Increment user count
        n_users += args.user_step


class BenchmarkClient:
    """Benchmark client class for capacity testing.

    A client that simulates a single user sending audio data to the A2F service
    and receiving facial expression animations. Supports WebSocket connections,
    chunked audio data transmission, and performance metrics collection.
    """

    _ssl_context_cache: None | ssl.SSLContext = None

    def __init__(
            self,
            host: str,
            port: int,
            disable_verify: bool,
            thread_pool_executor: ThreadPoolExecutor,
            sample_rate: int,
            channels: int,
            sample_width: int,
            audio_bytes: bytes,
            audio_duration: float,
            chunk_duration: float,
            timeout: float,
            ):
        """Initialize the benchmark client.

        Args:
            host (str): Server host address.
            port (int): Server port number.
            disable_verify (bool): Whether to disable SSL certificate verification.
            thread_pool_executor (ThreadPoolExecutor): Thread pool executor for
                executing synchronous operations.
            sample_rate (int): Audio sample rate in Hz.
            channels (int): Number of audio channels.
            sample_width (int): Audio sample width in bytes.
            audio_bytes (bytes): Byte sequence of audio data.
            audio_duration (float): Audio duration in seconds.
            chunk_duration (float): Audio chunk duration in seconds.
            timeout (float): Connection and operation timeout in seconds.
        """
        self.host: str = host
        self.port: str = port
        self.disable_verify: bool = disable_verify
        self.thread_pool_executor: ThreadPoolExecutor = thread_pool_executor
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.sample_width: int = sample_width
        self.audio_bytes: bytes = audio_bytes
        self.audio_duration = audio_duration
        self.chunk_duration: float = chunk_duration
        self.request_id: str | None = None
        self.audio_commit_time: float | None = None
        self.first_response_time: float | None = None
        self.final_response_time: float | None = None
        self.timeout: float = timeout
        self.state: Literal["idle", "running", "finished", "error"] = "idle"

    async def run(self):
        """Execute the main logic of the benchmark client.

        Establishes WebSocket connection, sends audio data in chunks, receives
        facial expression animation responses, and records key timestamps for
        performance metrics calculation.

        Execution flow:
        1. Establish WebSocket connection (supports WSS/WS protocols)
        2. Send audio start message
        3. Send audio data in chunks
        4. Send audio end message
        5. Receive and process response messages
        6. Record response timestamps
        """
        self.state = "running"
        try:
            loop = asyncio.get_running_loop()
            if self.port == 443:
                protocol = 'wss'
            else:
                protocol = 'ws'
            if not self.disable_verify or protocol == 'ws':
                ssl_context = None
            else:
                if self.__class__._ssl_context_cache is None:
                    ssl_context = await loop.run_in_executor(
                        self.thread_pool_executor, ssl.create_default_context)
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    self.__class__._ssl_context_cache = ssl_context
                else:
                    ssl_context = self.__class__._ssl_context_cache
            request_id = await loop.run_in_executor(
                self.thread_pool_executor, uuid.uuid4)
            self.request_id = str(request_id)
            async with websockets.connect(
                f"{protocol}://{self.host}:{self.port}/api/v1/streaming_audio2face/ws",
                ssl=ssl_context,
                ping_timeout=self.timeout,
                close_timeout=self.timeout,
            ) as websocket:
                chunk_start = a2f_v1_pb2.Audio2FaceBlendshapeV1Request()
                chunk_start.class_name = "StreamingAudio2FaceV1ChunkStart"
                chunk_start.request_id = self.request_id
                chunk_start.sample_rate = self.sample_rate
                chunk_start.sample_width = self.sample_width
                chunk_start.n_channels = self.channels
                chunk_start.profile_name = "KQ-default"
                pb_bytes = await loop.run_in_executor(
                    self.thread_pool_executor, chunk_start.SerializeToString)
                await websocket.send(pb_bytes)
                bytes_per_chunk = int(self.sample_rate * self.chunk_duration)
                for start_idx in range(0, len(self.audio_bytes), bytes_per_chunk):
                    end_idx = min(start_idx + bytes_per_chunk, len(self.audio_bytes))
                    if end_idx == start_idx:
                        break
                    chunk_body = a2f_v1_pb2.Audio2FaceBlendshapeV1Request()
                    chunk_body.class_name = "StreamingAudio2FaceV1ChunkBody"
                    chunk_body.pcm_bytes = self.audio_bytes[start_idx:end_idx]
                    pb_bytes = await loop.run_in_executor(
                        self.thread_pool_executor, chunk_body.SerializeToString)
                    await websocket.send(pb_bytes)
                chunk_end = a2f_v1_pb2.Audio2FaceBlendshapeV1Request()
                chunk_end.class_name = "StreamingAudio2FaceV1ChunkEnd"
                self.audio_commit_time = time.time()
                pb_bytes = await loop.run_in_executor(
                    self.thread_pool_executor, chunk_end.SerializeToString)
                await websocket.send(pb_bytes)
                first_response_received = False
                while True:
                    response_bytes = await asyncio.wait_for(
                        websocket.recv(), timeout=self.timeout)
                    response = a2f_v1_pb2.Audio2FaceBlendshapeV1Response()
                    await loop.run_in_executor(
                        self.thread_pool_executor,
                        response.ParseFromString,
                        response_bytes)
                    if response.class_name == "Audio2FaceV1ResponseChunkStart":
                        pass
                    elif response.class_name == "Audio2FaceV1ResponseChunkBody":
                        if not first_response_received:
                            self.first_response_time = time.time()
                            first_response_received = True
                    elif response.class_name == "Audio2FaceV1ResponseChunkEnd":
                        self.final_response_time = time.time()
                        self.state = "finished"
                        break
                    else:
                        raise ValueError(
                            f"Unknown response class name: {response.class_name}")
        except Exception as e:
            self.state = "error"
            raise e from e

    async def get_speed(self) -> float:
        """Calculate audio processing speed.

        Calculates the speed at which the A2F service processes audio,
        which is the ratio of audio duration to actual processing time.

        Returns:
            float: Audio processing speed in multiples of real-time
                (1.0 indicates real-time processing).

        Raises:
            ValueError: Raised when speed cannot be calculated, typically
                due to missing required timestamp records.
        """
        if self.final_response_time is None or self.audio_commit_time is None:
            raise ValueError(
                "Speed cannot be calculated, " +
                f"final_response_time={self.final_response_time}, " +
                f"audio_commit_time={self.audio_commit_time}, " +
                f"request_id={self.request_id}")
        return self.audio_duration / (self.final_response_time - self.audio_commit_time)

    async def get_latency(self) -> float:
        """Calculate first response latency.

        Calculates the time interval between sending the audio end message
        and receiving the first response message.

        Returns:
            float: First response latency in seconds.

        Raises:
            ValueError: Raised when latency cannot be calculated, typically
                due to missing required timestamp records.
        """
        if self.first_response_time is None or self.audio_commit_time is None:
            raise ValueError(
                "Latency cannot be calculated, " +
                f"first_response_time={self.first_response_time}, " +
                f"audio_commit_time={self.audio_commit_time}, " +
                f"request_id={self.request_id}")
        return self.first_response_time - self.audio_commit_time


if __name__ == "__main__":
    asyncio.run(main())
