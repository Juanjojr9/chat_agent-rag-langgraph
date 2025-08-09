# src/components/agent_builder.py
"""
Constructor del agente LangGraph + RAG
"""
import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI

from src.components.estado import State
from src.components.assistant import Assistant
from src.components.utils import create_tool_node_with_fallback, route_tools
from src.config.prompt import prompt as mi_prompt
from src.tools.Herramienta_RAG import Herramienta_RAG

logger = logging.getLogger(__name__)

class AgentBuilder:
    """Constructor del agente con configuración flexible."""
    
    def __init__(self, model: ChatOpenAI, tools: list = None):
        """
        Inicializa el constructor del agente.
        
        Args:
            model: Modelo de chat configurado
            tools: Lista de herramientas disponibles
        """
        self.model = model
        self.tools = tools or [Herramienta_RAG]
        self.prompt = mi_prompt
    
    def build(self) -> StateGraph:
        """
        Construye y devuelve el grafo del agente.
        
        Returns:
            StateGraph: Grafo del agente compilado
        """
        try:
            logger.info("Construyendo agente...")
            
            # Bind the tools to the LLM via the prompt
            assistant_runnable = self.prompt | self.model.bind_tools(self.tools)
            logger.debug(f"Herramientas configuradas: {[tool.name for tool in self.tools]}")

            # Build the state graph for the agent
            builder = StateGraph(State)
            
            # Define Nodes
            builder.add_node("assistant", Assistant(assistant_runnable, max_retries=3))
            builder.add_node("tools", create_tool_node_with_fallback(self.tools))

            # Define edges
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant", route_tools, ["tools", END]
            )
            builder.add_edge("tools", "assistant")

            # Set up the memory checkpointer for the graph
            memory = MemorySaver()
            agent_graph = builder.compile(
                checkpointer=memory,
            )
            logger.info("✅ Agente compilado exitosamente")
            return agent_graph
            
        except Exception as e:
            logger.exception("❌ Fallo inicializando el agente")
            raise

def build_agent(model: ChatOpenAI, tools: list = None) -> StateGraph:
    """
    Función de conveniencia para construir el agente.
    
    Args:
        model: Modelo de chat configurado
        tools: Lista de herramientas disponibles
        
    Returns:
        StateGraph: Grafo del agente compilado
    """
    builder = AgentBuilder(model, tools)
    return builder.build()