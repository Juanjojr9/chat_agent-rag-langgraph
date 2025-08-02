# 🧩 Agente LangGraph + RAG

Asistente conversacional en Python que combina:

| Componente | Tecnologías |
|------------|-------------|
| **Orquestación** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **Razonamiento LLM** | OpenAI GPT-4 (mini) vía `langchain_openai` |
| **Memoria de conocimiento** | RAG local con [FAISS](https://github.com/facebookresearch/faiss) |
| **Herramientas** | `Herramienta_RAG` (búsqueda semántica)<br>Fácil de extender con más tools |

> El CLI principal es **`chat_agente.py`**.  
> Los notebooks `prueba.ipynb` y `agente.ipynb` se usan solo para pruebas / exploración.

---

## 📂 Estructura rápida
```bash
├─ chat_agente.py # <— script interactivo que usarás en producción
├─ src/
│ ├─ components/
│ │ ├─ assistant.py
│ │ ├─ estado.py
│ │ └─ utils.py
│ ├─ config/
│ │ ├─ config.py
│ │ └─ prompt.py
│ └─ tools/
│ ├─ Herramienta_RAG.py
│ └─ rag.py
├─ data/ # corpus local; se crea índice FAISS aquí
├─ prueba.ipynb # notebook: carga corpus y genera índice
└─ agente.ipynb # notebook: ejemplo de flujo LangGraph
```


---

## 🚀 Instalación

```bash
# 1. Clona el repo
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# 2. Entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Dependencias
pip install -r requirements.txt
# o, si usas Poetry / pip-tools, adapta este paso


Variables de entorno
Crea un archivo .env en la raíz (o exports en tu shell):
OPENAI_API_KEY=sk-...
# Opcional: LANGCHAIN_TRACING_V2=true  (si usas LangSmith)
```

## 🗂️ Construir el índice RAG
Coloca tus PDFs / TXTs en la carpeta data/.

Abre el notebook prueba.ipynb y ejecuta las celdas:
esto lee los documentos, los parte en chunks, calcula embeddings y guarda:

data/index.faiss – vectores

data/metadata.jsonl – texto + metadatos

⚠️ Se usa CPU FAISS por defecto.
Si tienes CUDA, instala faiss-gpu y el script lo aprovechará automáticamente.


## 💬 Chatear con tu agente

```bash
python chat_agente.py
```
Escribe tu pregunta y pulsa Enter.

El agente consultará su RAG + GPT-4 y responderá.

Teclea exit o quit para salir.



## ⚙️ Personalización rápida

| Qué quieres cambiar |	Dónde | 
|------------|-------------|
| Modelo (p. ej. gpt-4o-2024-08-06) |	src/config/config.py → get_chat_model() | 
| Prompt, idioma, tono | src/config/prompt.py (función build_prompt si la activas) | 
| Añadir herramienta (requests, calculadora, etc.) | Crea un .py en src/tools/ y añádelo a la lista tools = [...] en chat_agente.py | 
| Tamaño del índice RAG | Ajusta el chunking / batch embeddings en src/tools/rag.py | 



## 🛠️ Solución de problemas
Mensaje	Causa & remedio
“Faiss GPU … not defined” ->	Aviso de que no tienes FAISS-GPU; ignóralo si usas CPU.
OPENAI_API_KEY not set  ->		Crea .env o exporta la variable en tu shell.
Se repite el texto del usuario	 ->	 Asegúrate de usar el último bloque for ev in agente.stream() dentro de chat_agente.py.
Consume demasiados tokens ->		Reduce max_tokens en get_chat_model() o recorta state["messages"] a los últimos n turnos.



## 📝 Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
