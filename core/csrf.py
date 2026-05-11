import secrets
import urllib.parse
from fastapi import Request, HTTPException
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
        if not request.url.path.startswith("/api"):
            return
        if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
            return

        cookie_token: Optional[str] = request.cookies.get(CSRF_TOKEN_NAME)
        header_token: Optional[str] = request.headers.get(CSRF_HEADER_NAME)

        if not cookie_token:
            raise HTTPException(status_code=403, detail="CSRF token missing from cookie")

        if header_token == cookie_token:
            return

        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            # Read body (sets _body so downstream _CachedRequest passes it on)
            try:
                body = await request.body()
                # Parse URL-encoded body to extract csrf_token
                params = urllib.parse.parse_qs(body.decode("latin-1"))
                token_list = params.get(CSRF_TOKEN_NAME, [])
                if token_list and token_list[0] == cookie_token:
                    return
            except Exception:
                pass
        elif "multipart/form-data" in content_type:
            # For multipart, read body and extract csrf_token from raw bytes
            # The csrf_token field is always appended last by JS, so find last match
            try:
                body = await request.body()
                raw = body.decode("utf-8", errors="replace")
                # Find all occurrences of name="csrf_token" followed by value
                import re
                token_field = re.escape(CSRF_TOKEN_NAME)
                # name="csrf_token"\r\n\r\nVALUE (multipart format)
                pattern = rf'name="{token_field}"\s*\r?\n\r?\n([^\r\n]+)'
                matches = list(re.finditer(pattern, raw))
                if matches:
                    form_token = matches[-1].group(1).strip()
                    if form_token == cookie_token:
                        return
            except Exception:
                pass

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
