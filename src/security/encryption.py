"""
Módulo de cifrado AES-256 para protección de datos sensibles.
Cifra/descifra archivos CSV y datos en tránsito.
"""

import os
import base64
import hashlib
import getpass
from pathlib import Path
from typing import Union, Optional
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from loguru import logger


class DataEncryptor:
    """Cifrador AES-256-CBC con autenticación HMAC-SHA256."""

    ITERATIONS = 100_000
    KEY_LENGTH = 32  # 256 bits
    SALT_LENGTH = 16
    IV_LENGTH = 16
    MAGIC = b"DENGUE_RD_v1"

    def __init__(self, password: Optional[str] = None):
        self._password = password or os.getenv("ENCRYPTION_KEY", "")
        if not self._password:
            raise ValueError("Se requiere contraseña de cifrado. Configure ENCRYPTION_KEY en .env")

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_LENGTH,
            salt=salt,
            iterations=self.ITERATIONS,
            backend=default_backend(),
        )
        return kdf.derive(self._password.encode("utf-8"))

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        salt = os.urandom(self.SALT_LENGTH)
        iv = os.urandom(self.IV_LENGTH)
        key = self._derive_key(salt)

        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()

        # HMAC para autenticidad
        h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(salt + iv + ciphertext)
        mac = h.finalize()

        return self.MAGIC + salt + iv + mac + ciphertext

    def decrypt_bytes(self, ciphertext_full: bytes) -> bytes:
        magic_len = len(self.MAGIC)
        if ciphertext_full[:magic_len] != self.MAGIC:
            raise ValueError("Archivo no es un archivo cifrado válido del sistema Dengue-RD")

        offset = magic_len
        salt = ciphertext_full[offset:offset + self.SALT_LENGTH]
        offset += self.SALT_LENGTH
        iv = ciphertext_full[offset:offset + self.IV_LENGTH]
        offset += self.IV_LENGTH
        mac = ciphertext_full[offset:offset + 32]
        offset += 32
        ciphertext = ciphertext_full[offset:]

        key = self._derive_key(salt)

        h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(salt + iv + ciphertext)
        h.verify(mac)  # lanza InvalidSignature si falla

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()

    def encrypt_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else input_path.with_suffix(".enc")

        logger.info(f"Cifrando: {input_path} → {output_path}")
        plaintext = input_path.read_bytes()
        encrypted = self.encrypt_bytes(plaintext)
        output_path.write_bytes(encrypted)
        logger.success(f"Archivo cifrado exitosamente: {output_path}")
        return output_path

    def decrypt_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else input_path.with_suffix("")

        logger.info(f"Descifrando: {input_path} → {output_path}")
        ciphertext = input_path.read_bytes()
        plaintext = self.decrypt_bytes(ciphertext)
        output_path.write_bytes(plaintext)
        logger.success(f"Archivo descifrado exitosamente: {output_path}")
        return output_path

    def encrypt_dataframe(self, df: pd.DataFrame) -> bytes:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return self.encrypt_bytes(csv_bytes)

    def decrypt_to_dataframe(self, encrypted: bytes) -> pd.DataFrame:
        import io
        csv_bytes = self.decrypt_bytes(encrypted)
        return pd.read_csv(io.BytesIO(csv_bytes))


def encrypt_file(input_path: str, output_path: Optional[str] = None, password: Optional[str] = None) -> str:
    enc = DataEncryptor(password)
    return str(enc.encrypt_file(input_path, output_path))


def decrypt_file(input_path: str, output_path: Optional[str] = None, password: Optional[str] = None) -> str:
    enc = DataEncryptor(password)
    return str(enc.decrypt_file(input_path, output_path))


def verify_password(input_password: str) -> bool:
    """Verifica contraseña de acceso a la aplicación."""
    stored = os.getenv("APP_PASSWORD", "")
    if not stored:
        logger.warning("APP_PASSWORD no configurado en .env")
        return False
    return hashlib.sha256(input_password.encode()).hexdigest() == \
           hashlib.sha256(stored.encode()).hexdigest()


def prompt_password(max_attempts: int = 3) -> bool:
    """Solicita contraseña al usuario por consola.
    En entornos no interactivos acepta TRAIN_PASSWORD como bypass."""
    auto = os.getenv("TRAIN_PASSWORD", "")
    if auto:
        if verify_password(auto):
            logger.info("Autenticación exitosa (TRAIN_PASSWORD)")
            return True
        logger.error("TRAIN_PASSWORD inválido")
        return False

    for attempt in range(1, max_attempts + 1):
        pwd = getpass.getpass(f"Contraseña de acceso [{attempt}/{max_attempts}]: ")
        if verify_password(pwd):
            logger.info("Autenticación exitosa")
            return True
        logger.warning(f"Contraseña incorrecta (intento {attempt}/{max_attempts})")
    logger.error("Acceso denegado — demasiados intentos fallidos")
    return False
