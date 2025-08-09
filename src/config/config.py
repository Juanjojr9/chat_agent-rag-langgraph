import os
import logging
import asyncio
from dotenv import load_dotenv
import nest_asyncio
import openai
from langchain_openai import ChatOpenAI
from langsmith.utils import tracing_is_enabled

logger = logging.getLogger(__name__)

# Configuración de asyncio para Windows
nest_asyncio.apply()

# Cargar variables de entorno
load_dotenv(override=True)
logger.info("Cargando .env…")

# API Keys - OpenAIModel will look for OPENAI_API_KEY in environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    logger.debug("OPENAI_API_KEY cargada correctamente")
    client = openai.OpenAI(api_key=api_key)
else:
   logger.critical("OPENAI_API_KEY no encontrada.")

# Configuración de LangSmith (opcional)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
logger.info("Tracing activo: %s", tracing_is_enabled())
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "mi_agente")

def configure_asyncio_policy():
    """Configura la política de asyncio para Windows."""
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.debug("Asyncio policy configurada para Windows.")

configure_asyncio_policy()
logger.info("Inicializando modelos OpenAI…")

def get_chat_model(
    name: str = "gpt-4o-mini",
    temperature: float = 0,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """Devuelve un ChatOpenAI ya configurado."""
    try:
        return ChatOpenAI(
            model=name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.exception("No se pudo inicializar el modelo %s: %s", name, str(e))
        raise

# Crea uno por defecto si realmente necesitas un global:
model = get_chat_model()