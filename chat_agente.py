#!/usr/bin/env python
"""
chat_agente.py  ·  CLI para conversar con tu agente LangGraph + RAG
-------------------------------------------------------------------
Ejecuta:
    python chat_agente.py
y escribe tu pregunta cuando veas el prompt 👤:
Sal con  exit  o  quit
"""

from __future__ import annotations

import logging
import sys
from typing import TypedDict

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,                  # cambia a DEBUG para ver más detalle
    stream=sys.stdout,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- IMPORTS DEL PROYECTO ----------
from src.config.config import get_chat_model
from src.config.prompt import prompt as mi_prompt
from src.components.utils import create_tool_node_with_fallback, route_tools, handle_tool_error,_print_event
from src.components.assistant import Assistant
from src.components.estado import State
from src.tools.Herramienta_RAG import Herramienta_RAG 
from src.tools.rag import init_rag


import os
import openai
from langchain.chains import LLMChain
from src.tools.rag import RAGLocal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel
from langgraph.graph import END



import logging, sys
logging.basicConfig(
    level=logging.DEBUG,               # o INFO
    stream=sys.stdout,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)



init_rag("data")
model = get_chat_model() 
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": "Prueba yy",
    }
}



# ---------- CONSTRUCTOR DEL AGENTE ----------
def build_agent() -> StateGraph:
    """
    Devuelve un grafo-agente:
    prompt  ➜  LLM  ➜  Assistant (con herramientas)  ➜  END
    """
    try:
        # Define the LLM for the agent
        llm = model #gpt-4.1-2025-04-14 / claude-3-5-sonnet-latest /gpt-4o-2024-08-06 /gpt-4.1-2025-04-14

        # Define the system prompt for the agent using a chat prompt template
        # Build the prompt template
        prompt = mi_prompt


        # Define the tools that the agent can use
        tools = [
            Herramienta_RAG
            ]

        # Bind the tools to the LLM via the prompt
        assistant_runnable = prompt | llm.bind_tools(tools )

        # Build the state graph for the agent
        #Initialize the Agent
        builder = StateGraph(State)
        #Define Nodes
        builder.add_node("assistant", Assistant(assistant_runnable, max_retries=3))
        builder.add_node("tools", create_tool_node_with_fallback(tools))

        #Define edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant", route_tools, ["tools", END]
        )
        builder.add_edge("tools", "assistant")




        # Set up the memory checkpointer for the graph
        memory = MemorySaver()
        agent_graph = builder.compile(
            checkpointer=memory,
            # NEW: The graph will always halt before executing the "sensitive_tools" node.
            # The user can approve or reject (or even alter the request) before
            # the assistant continues
            #interrupt_before=["sensitive_tools"],
        )
        logger.info("Agente compilado ✔️")
        return agent_graph
        
    except Exception as e:
        logger.exception("Fallo inicializando el agente")
        raise 
    



# ---------- BUCLE INTERACTIVO ----------
def main() -> None:
    agente = build_agent()
    state: State = {"messages": []}  
    _printed: set[str] = set()         # evita repetir eventos en _print_event

    print("\n💬  Escribe 'exit' para terminar.\n")

    while True:
        try:
            user_text = input("👤: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego!")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("👋 Hasta luego!")
            break
        if not user_text:
            continue

        # 1) turno del usuario
        state["messages"].append(("user", user_text))

        # 2) stream del grafo  ──────────────────────────────────────────
        for ev in agente.stream(state, config, stream_mode="values"):
            state = ev                      # ← cada 'ev' es el nuevo estado
            # _print_event(str(ev), _printed)   # opcional, si tu helper acepta texto

        # 3) extrae la última respuesta del asistente
        asistencia = None
        for msg in reversed(state["messages"]):
            # LangChain guarda HumanMessage / AIMessage; tú añades tuplas
            if getattr(msg, "type", None) == "ai" or (
                isinstance(msg, tuple) and msg[0] == "assistant"
            ):
                asistencia = msg.content if hasattr(msg, "content") else msg[1]
                break

        if asistencia:
            print(f"🤖: {asistencia}\n")
        else:
            logger.warning("No encontré respuesta del asistente; quizá falló la herramienta.")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    main()