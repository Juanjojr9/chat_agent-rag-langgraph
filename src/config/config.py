import os
from dotenv import load_dotenv; 
import nest_asyncio, asyncio
from langchain.chat_models import init_chat_model
import openai
from langchain_openai import ChatOpenAI
from langsmith.utils import tracing_is_enabled
import nest_asyncio, asyncio, logging


logger = logging.getLogger(__name__)

nest_asyncio.apply()

load_dotenv(override=True)
logger.info("Cargando .env…")

# API Keys - OpenAIModel will look for OPENAI_API_KEY in environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    logger.debug("OPENAI_API_KEY cargada (longitud %d)", len(api_key))
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
   logger.critical("OPENAI_API_KEY no encontrada.")


os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
logger.info("Tracing activo: %s", tracing_is_enabled())
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

def configure_asyncio_policy():
    if os.name == 'nt':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.debug("Asyncio policy configurada para Windows.")


configure_asyncio_policy()
logger.info("Inicializando modelos OpenAI…")


def get_chat_model(
    name: str = "gpt-4.1-mini",
    temperature: float = 0,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """Devuelve un ChatOpenAI ya configurado."""
    try:
        return ChatOpenAI(model=name,
                          temperature=temperature,
                          max_tokens=max_tokens)
    except Exception:
        logger.exception("No se pudo inicializar el modelo %s", name)
        raise

# Crea uno por defecto si realmente necesitas un global:
model = get_chat_model()