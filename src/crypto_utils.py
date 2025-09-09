import base64
import os
import time
from typing import Any, Dict

import jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import settings


def generate_jwt(payload: Dict[str, Any], expires_in_seconds: int = 3600) -> str:
    to_encode = payload.copy()
    to_encode["exp"] = int(time.time()) + expires_in_seconds
    token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token


def verify_jwt(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


def _derive_key(secret: str) -> bytes:
    # Derive 32-byte key from secret using simple hash; replace with HKDF in production
    import hashlib

    return hashlib.sha256(secret.encode("utf-8")).digest()


def encrypt_bytes(plaintext: bytes, secret: str) -> bytes:
    key = _derive_key(secret)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt_bytes(ciphertext_with_nonce: bytes, secret: str) -> bytes:
    key = _derive_key(secret)
    aesgcm = AESGCM(key)
    nonce = ciphertext_with_nonce[:12]
    ciphertext = ciphertext_with_nonce[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)


def encrypt_text(plaintext: str, secret: str) -> str:
    return base64.b64encode(encrypt_bytes(plaintext.encode("utf-8"), secret)).decode("utf-8")


def decrypt_text(ciphertext_b64: str, secret: str) -> str:
    return decrypt_bytes(base64.b64decode(ciphertext_b64.encode("utf-8")), secret).decode("utf-8")


