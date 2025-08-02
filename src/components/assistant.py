
from langgraph.graph import StateGraph, END

from langchain_core.runnables import Runnable, RunnableConfig
from src.components.estado import State
from typing import Dict, Any


class Assistant:
    def __init__(self, runnable: Runnable, max_retries: int = 3) -> None:
        """Wrapper para ejecutar un runnable LangChain con re‑intentos."""
        self.runnable: Runnable = runnable
        self.max_retries: int = max_retries

    def __call__(self, state: State, config: RunnableConfig | None = None):
        retries = 0
        while retries < self.max_retries:
                result = self.runnable.invoke(state, config)
                if self._needs_retry(result):
                    retries += 1
                    # Insertamos un aviso al final del historial
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                    continue
                return {"messages": result}
       # Si llegamos aquí no hubo respuesta válida
        raise RuntimeError("Assistant agotó los re‑intentos sin obtener salida.")

    @staticmethod
    def _needs_retry(result: Any) -> bool:
        """Evalúa si la salida del LLM está vacía o es irrelevante."""
        if getattr(result, "tool_calls", None):
            return False
        if not getattr(result, "content", None):
            return True
        if isinstance(result.content, list) and not result.content[0].get("text"):
            return True
        return False