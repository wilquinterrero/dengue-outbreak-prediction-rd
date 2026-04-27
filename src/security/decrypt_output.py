"""
Utilidad de línea de comandos para descifrar archivos de salida.
Uso:  python src/security/decrypt_output.py <archivo.enc> [salida.csv]

Los archivos .enc son los outputs cifrados generados por el sistema
para Power BI y análisis externo.
"""

import sys
import getpass
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    if len(sys.argv) < 2:
        print("Uso: python decrypt_output.py <archivo.enc> [archivo_salida.csv]")
        print("\nArchivos disponibles en outputs/:")
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            for f in outputs_dir.glob("*.enc"):
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Archivo no encontrado: {input_path}")
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".csv")

    password = getpass.getpass("Contraseña de cifrado: ")

    try:
        from src.security.encryption import DataEncryptor
        enc = DataEncryptor(password=password)
        enc.decrypt_file(input_path, output_path)
        print(f"\n✓ Archivo descifrado exitosamente: {output_path}")
    except Exception as e:
        print(f"\n✗ Error al descifrar: {e}")
        print("  Verifique que la contraseña sea correcta y que el archivo sea válido.")
        sys.exit(1)


if __name__ == "__main__":
    main()
