#!/usr/bin/env python
"""
chat_agente.py  ·  CLI para conversar con tu agente LangGraph + RAG
-------------------------------------------------------------------
Ejecuta:
    python chat_agente.py
y escribe tu pregunta cuando veas el prompt ��:
Sal con  exit  o  quit
"""

from __future__ import annotations

import sys
import os
from typing import TypedDict

# ---------- CONFIGURACIÓN DE LOGGING ----------
from src.config.logging_config import setup_logging, get_logger
import os
from datetime import datetime

# Crear carpeta de logs si no existe
os.makedirs("logs", exist_ok=True)

# Generar nombre de archivo con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/mi_agente_{timestamp}.log"

# Configurar logging antes de importar otros módulos
setup_logging(
    level="INFO", #INFO #DEBUG
    log_file=log_filename,
    format_string="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = get_logger(__name__)

# ---------- IMPORTS DEL PROYECTO ----------
from src.config.config import get_chat_model
from src.components.estado import State
from src.components.agent_builder import build_agent  # ← NUEVO IMPORT
from src.tools.rag import init_rag

# ---------- CONFIGURACIÓN ----------
logger.info("Inicializando RAG...")
init_rag("data")
logger.info("Inicializando modelo de chat...")
model = get_chat_model() 

# Configuración dinámica del thread_id
def get_thread_id() -> str:
    """Genera un ID único para la sesión actual."""
    import uuid
    return f"session_{uuid.uuid4().hex[:8]}"

config = {
    "configurable": {
        "thread_id": get_thread_id(),
    }
}

# ---------- BUCLE INTERACTIVO ----------
def main() -> None:
    try:
        # Construir el agente usando el nuevo componente
        agente = build_agent(model)
        state: State = {"messages": []}  
        _printed: set[str] = set()

        print("\n💬  Escribe 'exit' para terminar.\n")
        logger.info("🚀 Agente listo para conversar")

        while True:
            try:
                user_text = input("👤: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Hasta luego!")
                logger.info("Sesión terminada por el usuario")
                break

            if user_text.lower() in {"exit", "quit"}:
                print("👋 Hasta luego!")
                logger.info("Sesión terminada por comando")
                break
            if not user_text:
                continue

            logger.debug(f"Usuario dice: {user_text[:50]}...")
            
            # 1) turno del usuario
            state["messages"].append(("user", user_text))

            # 2) stream del grafo
            try:
                for ev in agente.stream(state, config, stream_mode="values"):
                    state = ev
                logger.debug("Procesamiento del agente completado")
            except Exception as e:
                logger.error(f"Error en el procesamiento del agente: {str(e)}")
                print(f"🤖: Lo siento, hubo un error procesando tu pregunta: {str(e)}")
                continue

            # 3) extrae la última respuesta del asistente
            asistencia = None
            for msg in reversed(state["messages"]):
                if getattr(msg, "type", None) == "ai" or (
                    isinstance(msg, tuple) and msg[0] == "assistant"
                ):
                    asistencia = msg.content if hasattr(msg, "content") else msg[1]
                    break

            if asistencia:
                print(f"��: {asistencia}\n")
                logger.debug(f"Asistente responde: {asistencia[:100]}...")
            else:
                logger.warning("No se encontró respuesta del asistente")
                print("🤖: Lo siento, no pude generar una respuesta válida.")

    except Exception as e:
        logger.exception("Error crítico en la aplicación")
        print(f"❌ Error crítico: {str(e)}")
        sys.exit(1)

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    main()