import os
from dotenv import load_dotenv
from pathlib import Path

def find_dotenv(start_path='.'):
    """Busca el archivo .env ascendiendo desde la ruta de inicio."""
    current_path = Path(start_path).resolve()
    while current_path != current_path.parent:  # Evita el directorio raíz
        if (current_path / '.env').is_file():
            return str(current_path / '.env')
        current_path = current_path.parent
    return None

# Prueba de carga del archivo .env

if __name__ == '__main__':

    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
        model_path = os.getenv("MODEL")
        print(f"Ruta del modelo cargada desde: {dotenv_path}")
        print(f"MODEL: {model_path}")
    else:
        print("No se encontró el archivo .env.")