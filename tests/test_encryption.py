"""
Tests unitarios para el módulo de cifrado AES-256.
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def encryptor():
    from src.security.encryption import DataEncryptor
    return DataEncryptor(password="test-password-for-unit-tests!")


class TestDataEncryptor:
    def test_encrypt_decrypt_bytes_roundtrip(self, encryptor):
        """El texto descifrado debe ser idéntico al original."""
        plaintext = b"Datos de dengue: Santo Domingo, semana 15, 127 casos"
        encrypted = encryptor.encrypt_bytes(plaintext)
        decrypted = encryptor.decrypt_bytes(encrypted)
        assert decrypted == plaintext

    def test_encrypted_different_from_plaintext(self, encryptor):
        plaintext = b"texto de prueba del sistema dengue rd"
        encrypted = encryptor.encrypt_bytes(plaintext)
        assert encrypted != plaintext

    def test_magic_header_present(self, encryptor):
        encrypted = encryptor.encrypt_bytes(b"test data")
        assert encrypted.startswith(b"DENGUE_RD_v1")

    def test_different_encryptions_same_plaintext(self, encryptor):
        """Cada cifrado usa IV aleatorio, por lo que produce bytes distintos."""
        plain = b"mismo texto"
        enc1 = encryptor.encrypt_bytes(plain)
        enc2 = encryptor.encrypt_bytes(plain)
        assert enc1 != enc2  # IV diferente cada vez

    def test_wrong_password_raises(self):
        from src.security.encryption import DataEncryptor
        from cryptography.exceptions import InvalidSignature
        enc_good = DataEncryptor(password="correct-password-test")
        enc_bad  = DataEncryptor(password="wrong-password-test!")
        encrypted = enc_good.encrypt_bytes(b"secret")
        with pytest.raises((InvalidSignature, ValueError, Exception)):
            enc_bad.decrypt_bytes(encrypted)

    def test_tampered_data_raises(self, encryptor):
        from cryptography.exceptions import InvalidSignature
        encrypted = encryptor.encrypt_bytes(b"datos originales")
        tampered = bytearray(encrypted)
        tampered[50] ^= 0xFF  # modificar un byte
        with pytest.raises((InvalidSignature, ValueError, Exception)):
            encryptor.decrypt_bytes(bytes(tampered))

    def test_invalid_magic_raises(self, encryptor):
        with pytest.raises(ValueError, match="no es un archivo cifrado válido"):
            encryptor.decrypt_bytes(b"esto no es un archivo cifrado valido")

    def test_encrypt_file(self, encryptor, tmp_path):
        """Cifra un archivo y verifica que el archivo cifrado existe y es distinto."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("province,cases\nSantiago,100\nAzua,50")
        enc_path = encryptor.encrypt_file(str(test_file))
        assert Path(enc_path).exists()
        assert Path(enc_path).read_bytes() != test_file.read_bytes()

    def test_decrypt_file_roundtrip(self, encryptor, tmp_path):
        """Cifra y descifra un archivo CSV — debe coincidir exactamente."""
        original = "province,cases\nSanto Domingo,200\nSantiago,150\n"
        test_file = tmp_path / "data.csv"
        test_file.write_text(original)
        enc_path = encryptor.encrypt_file(str(test_file))
        dec_path = tmp_path / "decrypted.csv"
        encryptor.decrypt_file(str(enc_path), str(dec_path))
        assert dec_path.read_text() == original

    def test_encrypt_decrypt_dataframe(self, encryptor):
        """DataFrame cifrado debe recuperarse sin pérdida de datos."""
        df_original = pd.DataFrame({
            "province": ["Santo Domingo", "Santiago"],
            "risk_index": [72.5, 45.3],
            "cases": [150, 80],
        })
        encrypted = encryptor.encrypt_dataframe(df_original)
        df_recovered = encryptor.decrypt_to_dataframe(encrypted)
        pd.testing.assert_frame_equal(df_original, df_recovered)

    def test_no_password_raises(self):
        from src.security.encryption import DataEncryptor
        with patch.dict(os.environ, {"ENCRYPTION_KEY": ""}):
            with pytest.raises(ValueError, match="Se requiere contraseña"):
                DataEncryptor(password="")


class TestPasswordVerification:
    def test_correct_password(self):
        from src.security.encryption import verify_password
        with patch.dict(os.environ, {"APP_PASSWORD": "mi_password_seguro"}):
            assert verify_password("mi_password_seguro") is True

    def test_incorrect_password(self):
        from src.security.encryption import verify_password
        with patch.dict(os.environ, {"APP_PASSWORD": "correcto"}):
            assert verify_password("incorrecto") is False

    def test_empty_stored_password(self):
        from src.security.encryption import verify_password
        with patch.dict(os.environ, {"APP_PASSWORD": ""}):
            assert verify_password("cualquier") is False
