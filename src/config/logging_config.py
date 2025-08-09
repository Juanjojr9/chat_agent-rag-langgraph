import logging
import sys
from typing import Optional

def setup_logging(
    level: str = "INFO",           # Nivel de logging
    log_file: Optional[str] = None, # Archivo para guardar logs
    format_string: Optional[str] = None # Formato personalizado
) -> None:
    """
    Configura el logging para toda la aplicación.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Archivo donde guardar los logs (opcional)
        format_string: Formato personalizado para los logs (opcional)
    """
    
    # Nivel de logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Formato por defecto
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    # Configurar el logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Limpiar handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Handler para archivo (si se especifica)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging configurado: nivel={level}, archivo={log_file}")
        except Exception as e:
            logging.error(f"No se pudo configurar logging a archivo {log_file}: {e}")
    
    # Configurar niveles específicos para librerías externas
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    
    logging.info(f"Logging configurado: nivel={level}")

def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado para un módulo específico.
    
    Args:
        name: Nombre del módulo (ej: __name__)
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)
