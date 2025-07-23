from langchain_core.tools import tool
from typing import Dict
import openai
import os
from src.tools.rag import RAGLocal
from typing import Annotated

from src.tools.rag import init_rag


@tool
def Herramienta_RAG(
    input: str,
    k: int = 1) -> str:
    """Siempre usa esta herramienta Esta herramienta te permite buscar información sobre:
        redes neuronales
        deep learning
        inteligencia artificial
        machine learning
        aprendizaje profundo
        aprendizaje automático


    Args:
        input: el texto para realizar la busqueda en la base de datos vecotrizada
        k: el número de trozos que recuperamos de la base de datos vectorizada

    """
    rag = init_rag("data")      # no hace nada si ya existe
    return rag.query(input, k=k)