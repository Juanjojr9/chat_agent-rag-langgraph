
from langgraph.graph import StateGraph, END

from langchain_core.runnables import Runnable, RunnableConfig
from src.components.estado import State
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, runnable: Runnable, max_retries: int = 3) -> None:
        """Wrapper para ejecutar un runnable LangChain con re‑intentos."""
        self.runnable: Runnable = runnable
        self.max_retries: int = max_retries

    def __call__(self, state: State, config: RunnableConfig | None = None):
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                logger.debug(f"Intento {retries + 1}/{self.max_retries} del asistente")
                result = self.runnable.invoke(state, config)
                
                if self._needs_retry(result):
                    retries += 1
                    logger.warning(f"Respuesta vacía o inválida, reintentando... (intento {retries})")
                    # Insertamos un aviso al final del historial
                    messages = state["messages"] + [("user", "Por favor, proporciona una respuesta válida y útil.")]
                    state = {**state, "messages": messages}
                    continue
                
                logger.debug("Respuesta del asistente obtenida exitosamente")
                return {"messages": result}
                
            except Exception as e:
                last_error = e
                retries += 1
                logger.error(f"Error en intento {retries}: {str(e)}")
                
                if retries < self.max_retries:
                    # Añadir mensaje de error al estado para el siguiente intento
                    messages = state["messages"] + [("user", f"Hubo un error: {str(e)}. Por favor, intenta de nuevo.")]
                    state = {**state, "messages": messages}
                else:
                    logger.error("Se agotaron todos los reintentos")
                    break
        
        # Si llegamos aquí no hubo respuesta válida
        error_msg = f"Assistant agotó los re‑intentos sin obtener salida válida. Último error: {str(last_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @staticmethod
    def _needs_retry(result: Any) -> bool:
        """Evalúa si la salida del LLM está vacía o es irrelevante."""
        try:
            # Verificar si tiene tool_calls (es válido)
            if getattr(result, "tool_calls", None):
                return False
            
            # Verificar si tiene contenido
            if not getattr(result, "content", None):
                return True
            
            # Verificar si el contenido es una lista vacía o sin texto
            if isinstance(result.content, list):
                if not result.content:
                    return True
                # Verificar si el primer elemento tiene texto
                if result.content and not result.content[0].get("text"):
                    return True
            
            # Verificar si el contenido es una cadena vacía
            if isinstance(result.content, str) and not result.content.strip():
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error al evaluar resultado: {str(e)}")
            return True