from fastapi.responses import JSONResponse
from typing import Any, Optional, Dict


def success_response(data: Any = None, message: str = "操作成功", code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "success": True,
            "message": message,
            "data": data
        }
    )


def error_response(message: str = "操作失败", details: Optional[Dict] = None, code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "success": False,
            "message": message,
            "details": details or {}
        }
    )
