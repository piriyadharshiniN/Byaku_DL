from typing import Any, Optional, Dict
from fastapi.responses import JSONResponse
from fastapi import Request
from backend.exceptions import ByakuException

def api_success(data: Any = None, status_code: int = 200) -> JSONResponse:
    content = {
        "ok": True,
        "data": data,
        "error": None
    }
    return JSONResponse(status_code=status_code, content=content)

def api_error(code: str, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    err_obj = {
        "code": code,
        "message": message
    }
    if details:
        err_obj["details"] = details
    content = {
        "ok": False,
        "data": None,
        "error": err_obj
    }
    return JSONResponse(status_code=status_code, content=content)

async def byaku_exception_handler(request: Request, exc: ByakuException) -> JSONResponse:
    return api_error(code=exc.code, message=exc.message, status_code=exc.status_code, details=exc.details)

async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # No stack traces leaked to clients
    return api_error(code="INTERNAL_SERVER_ERROR", message="An unexpected internal server error occurred.", status_code=500)
