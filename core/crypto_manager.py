import base64
import os

from cryptography.fernet import Fernet


def _load_key() -> bytes:
    env_key = os.getenv("FACE_DATA_KEY")
    if env_key:
        key_bytes = env_key.encode("utf-8")
        if len(key_bytes) >= 32:
            return key_bytes

    key_path = os.path.join("secrets", "face_data.key")
    if os.path.exists(key_path):
        try:
            with open(key_path, "rb") as f:
                key_data = f.read().strip()
                if len(key_data) >= 32:
                    return key_data
        except Exception:
            pass

    raise RuntimeError("FACE_DATA_KEY not configured or invalid")


def encrypt_bytes(data: bytes) -> bytes:
    f = Fernet(_load_key())
    return f.encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    f = Fernet(_load_key())
    return f.decrypt(token)


def encrypt_to_b64(data: bytes) -> str:
    return base64.b64encode(encrypt_bytes(data)).decode("utf-8")


def decrypt_from_b64(token_b64: str) -> bytes:
    return decrypt_bytes(base64.b64decode(token_b64.encode("utf-8")))

