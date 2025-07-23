from langchain.prompts.chat import ChatPromptTemplate
from src.tools.rag import RAGLocal

def modelo_rag(
    query: str,
    rag_client: RAGLocal,
    prompt_template: ChatPromptTemplate,
    chat_model,
    k: int = 3
) -> str:
    """
    Realiza una consulta RAG y envía al modelo la pregunta con contexto.

    Parámetros:
    - query: texto de la pregunta del usuario.
    - rag_client: instancia de RAGDrive ya inicializada (y con índice FAISS creado).
    - prompt_template: ChatPromptTemplate parcialmente configurado (time y client).
    - chat_model: modelo de chat inicializado (p.ej. gpt-4.1-2025-04-14).
    - k: número de documentos a recuperar para contexto (por defecto 3).

    Devuelve:
    - La respuesta generada por el modelo de chat.
    """
    # 1. Recuperar los k fragmentos más relevantes
    retrieved_text = rag_client.query(query, k=k)
    print(retrieved_text)
    # 2. Construir el mensaje humano incluyendo el contexto
    #    Podrías adaptar el prefijo “Contexto relevante” al estilo que prefieras.
    human_input = (
        f"Contexto relevante:\n{retrieved_text}\n\n"
        f"Usuario pregunta: {query}"
    )

    # 3. Formatear el prompt completo (ya incluye SystemMessage con time/client)
    prompt_value = prompt_template.format_prompt(messages=human_input)
    messages = prompt_value.to_messages()  # lista de mensajes para el chat

    # 4. Invocar al modelo y devolver la respuesta
    return chat_model.invoke(messages)
