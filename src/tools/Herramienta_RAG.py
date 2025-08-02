from langchain_core.tools import tool
from typing import Dict
import openai
import os
from src.tools.rag import RAGLocal
from typing import Annotated

from src.tools.rag import init_rag

_rag = init_rag("data") 
@tool
def Herramienta_RAG(
    input: str,
    k: int = 1) -> str:
    """Devuelve la búsqueda RAG para *input* recuperando *k* fragmentos. Esta herramienta te permite buscar información sobre:
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
 
    return _rag.query(input, k=k)