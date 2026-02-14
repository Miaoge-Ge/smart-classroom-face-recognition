import secrets
from fastapi import Request, HTTPException
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

CSRF_TOKEN_NAME = "csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"


class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            await self._validate_csrf(request)
        response = await call_next(request)
        return response

    async def _validate_csrf(self, request: Request):
        if request.url.path.startswith("/api") and request.method in ("POST", "PUT", "DELETE", "PATCH"):
            form_token: Optional[str] = None
            header_token: Optional[str] = None
            cookie_token: Optional[str] = request.cookies.get(CSRF_TOKEN_NAME)

            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
                try:
                    form = await request.form()
                    form_token = form.get(CSRF_TOKEN_NAME)
                except Exception:
                    pass

            header_token = request.headers.get(CSRF_HEADER_NAME)

            if not cookie_token:
                raise HTTPException(status_code=403, detail="CSRF token missing from cookie")

            if not (form_token == cookie_token or header_token == cookie_token):
                raise HTTPException(status_code=403, detail="CSRF token validation failed")


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response, token: str):
    response.set_cookie(
        key=CSRF_TOKEN_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=3600 * 24,
    )
    return token
