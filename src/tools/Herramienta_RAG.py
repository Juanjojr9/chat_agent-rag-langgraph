from langchain_core.tools import tool
from typing import Dict
import openai
import os
from src.tools.rag import RAGLocal
from typing import Annotated
import logging

from src.tools.rag import init_rag

logger = logging.getLogger(__name__)

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
        input: el texto para realizar la busqueda en la base de datos vectorizada
        k: el número de trozos que recuperamos de la base de datos vectorizada (1-10)

    Returns:
        str: Fragmentos relevantes encontrados en la base de datos
    """
    
    # Validación de entrada
    if not input or not input.strip():
        return "Error: El texto de búsqueda no puede estar vacío."
    
    if not isinstance(k, int) or k < 1 or k > 10:
        return "Error: El parámetro 'k' debe ser un número entero entre 1 y 10."
    
    try:
        # Limpiar y normalizar el input
        clean_input = input.strip()
        logger.debug(f"Realizando búsqueda RAG para: '{clean_input}' con k={k}")
        
        result = _rag.query(clean_input, k=k)
        
        if not result or result.strip() == "":
            return "No se encontraron documentos relevantes para tu búsqueda."
        
        return result
        
    except Exception as e:
        logger.error(f"Error en búsqueda RAG: {str(e)}")
        return f"Error interno en la búsqueda: {str(e)}"