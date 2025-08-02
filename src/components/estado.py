from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    """Estructura de datos que mantiene el historial de la conversación."""
    messages: Annotated[list[AnyMessage], add_messages]
