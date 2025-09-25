import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..utils.log import get_logger


class APIErrorMessage(BaseModel):
    """API error message model.

    Standardized error response format for API endpoints.
    """
    message: str
    code: int
    detail: dict | None = None


OPENAPI_RESPONSE_400 = {
    '400': {
        'description': 'Bad Request',
        'model': APIErrorMessage
    }
}
OPENAPI_RESPONSE_401 = {
    '401': {
        'description': 'Invalid Authentication',
        'model': APIErrorMessage
    }
}
OPENAPI_RESPONSE_403 = {
    '403': {
        'description': 'Forbidden',
        'model': APIErrorMessage
    }
}
OPENAPI_RESPONSE_404 = {
    '404': {
        'description': 'Not found',
        'model': APIErrorMessage
    }
}
OPENAPI_RESPONSE_422 = {
    '422': {
        'description': 'Validaion Error',
        'model': APIErrorMessage
    }
}
OPENAPI_RESPONSE_500 = {
    '500': {
        'description': 'Internal Server Error',
        'model': APIErrorMessage
    }
}
OPENAPI_RESPONSE_501 = {
    '501': {
        'description': 'Not Implemented',
        'model': APIErrorMessage
    }
}
OPENAPI_RESPONSE_503 = {
    '503': {
        'description': 'Service Unavailable',
        'model': APIErrorMessage
    }
}

class NoLogFileException(HTTPException):
    """Exception raised when no log file is found."""
    pass

async def http_exception_handler(request, exc):
    """Handle HTTP exceptions and return standardized error responses.

    Args:
        request:
            The incoming request object.
        exc:
            The HTTP exception that was raised.

    Returns:
        Response:
            JSON response with standardized error format or simple response
            for status codes that don't allow body content.
    """
    headers = getattr(exc, 'headers', None)
    if headers is None:
        headers = {}
    if request.headers is not None:
        headers = {**headers, **request.headers}

    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=headers)

    content = APIErrorMessage(
        message=f'HTTP Error {exc.status_code}',
        code=exc.status_code,
        detail={'error': exc.detail},
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=content.model_dump(),
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc):
    """Handle request validation errors and return standardized error responses.

    Args:
        request (Request):
            The incoming request object.
        exc:
            The validation exception that was raised.

    Returns:
        JSONResponse:
            JSON response with validation error details.
    """
    content = APIErrorMessage(
        message='Validaion Error',
        code=422,
        detail={
            'errors': exc.errors(),
            'body': exc.body
        },
    )

    return JSONResponse(
        content=content.model_dump(),
        status_code=422,
        headers=request.headers,
    )


async def exception_handler(request: Request, exc):
    """Handle general exceptions and return standardized error responses.

    Logs the full exception traceback and returns a generic internal
    server error response.

    Args:
        request (Request):
            The incoming request object.
        exc:
            The exception that was raised.

    Returns:
        JSONResponse:
            JSON response with internal server error details.
    """
    content = APIErrorMessage(
        message='Internal Server Error',
        code=500,
        detail={'error': f'{exc}, type: {type(exc)}'},
    )
    logger = get_logger()
    logger.error(traceback.format_exc())
    return JSONResponse(
        content=content.model_dump(),
        status_code=500,
        headers=request.headers,
    )


def register_error_handlers(app: FastAPI):
    """Register error handlers for the FastAPI application.

    Registers handlers for HTTP exceptions, validation errors, and general
    exceptions to provide standardized error responses.

    Args:
        app (FastAPI):
            The FastAPI application instance to register handlers with.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError,
                              validation_exception_handler)
    app.add_exception_handler(Exception, exception_handler)
