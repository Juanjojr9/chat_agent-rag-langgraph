# 🧩 Agente LangGraph + RAG

Asistente conversacional inteligente en Python que combina LangGraph con RAG local para crear un agente conversacional potente y personalizable.

## ✨ Características

- 🤖 **Agente conversacional inteligente** con LangGraph
- 📚 **RAG local** con FAISS para búsqueda semántica
- 🛠️ **Arquitectura modular** fácil de extender
- 📊 **Sistema de logging** profesional con archivos de log
- 🔄 **Re-intentos automáticos** con manejo de errores robusto
- 🎨 **Respuestas poéticas** en español
- 📄 **Procesamiento de documentos** (PDF, DOCX, TXT)
- 🧵 **Thread IDs dinámicos** para gestión de sesiones
- 🎯 **Búsqueda semántica** avanzada en documentos
- ⚡ **Performance optimizada** con batching de embeddings

## 🏗️ Arquitectura

| Componente | Tecnologías |
|------------|-------------|
| **Orquestación** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **Razonamiento LLM** | OpenAI GPT-4 (mini) vía `langchain_openai` |
| **Memoria de conocimiento** | RAG local con [FAISS](https://github.com/facebookresearch/faiss) |
| **Herramientas** | `Herramienta_RAG` (búsqueda semántica)<br>Fácil de extender con más tools |
| **Logging** | Sistema centralizado con archivos timestamped |
| **Configuración** | Variables de entorno + archivos de configuración |

> **💡 El CLI principal es `chat_agente.py`**  
> Los notebooks `prueba.ipynb` y `agente.ipynb` se usan solo para pruebas / exploración.

---

## 📂 Estructura del Proyecto

```
mi_agente/
├── chat_agente.py              # 🚀 Script principal para usar en producción
├── src/
│   ├── components/
│   │   ├── agent_builder.py    # 🔧 Constructor del agente
│   │   ├── assistant.py        # 🤖 Wrapper del LLM con retry logic
│   │   ├── estado.py           # 📊 Definición del estado LangGraph
│   │   └── utils.py            # 🛠️ Utilidades del agente
│   ├── config/
│   │   ├── config.py           # ⚙️ Configuración del modelo
│   │   ├── logging_config.py   # 📝 Sistema de logging centralizado
│   │   └── prompt.py           # 💬 Prompts del sistema
│   └── tools/
│       ├── Herramienta_RAG.py  # 🔍 Herramienta de búsqueda semántica
│       ├── rag.py              # 📚 Implementación RAG local
│       └── rag_promp.py        # 📝 Prompts para RAG
├── data/                       # 📁 Documentos y índices FAISS
│   ├── faiss_indexes/          # 🔍 Índices vectoriales
│   └── *.pdf, *.txt, *.docx    # 📄 Documentos para procesar
├── logs/                       # 📊 Archivos de log timestamped
├── requirements.txt            # 📦 Dependencias del proyecto
├── README.md                   # 📖 Este archivo
├── prueba.ipynb               # 🧪 Notebook para crear índices RAG
└── agente.ipynb               # 🔬 Notebook de experimentación
```

---

## 📋 Requisitos del Sistema

- **Python**: 3.9 o superior
- **Memoria RAM**: Mínimo 4GB (recomendado 8GB+)
- **Espacio en disco**: 1GB para dependencias + espacio para documentos
- **API Key**: OpenAI API key válida
- **Sistema operativo**: Windows, macOS, Linux

---

## 🚀 Instalación Rápida

### 1. Clonar y Preparar
```bash
# Clona el repositorio
git clone https://github.com/tu-usuario/mi_agente.git
cd mi_agente

# Crear entorno virtual (recomendado)
python -m venv .venv

# Activar entorno virtual
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Instalar Dependencias
```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

### 3. Configurar Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto:

```bash
# Variables obligatorias
OPENAI_API_KEY=sk-your-openai-api-key-here

# Variables opcionales
MODEL_NAME=gpt-4o-mini
MODEL_TEMPERATURE=0
MODEL_MAX_TOKENS=1024
LOG_LEVEL=INFO
RAG_CHUNK_SIZE=1000
RAG_OVERLAP=25
```

---

## 🗂️ Preparar el Índice RAG

### 1. Añadir Documentos
Coloca tus documentos en la carpeta `data/`:
- **PDFs**: `.pdf`
- **Documentos Word**: `.docx`
- **Archivos de texto**: `.txt`

### 2. Crear el Índice
```bash
# Opción A: Usar el notebook
jupyter lab prueba.ipynb

# Opción B: Usar Python directamente
python -c "
from src.tools.rag import RAG
rag = RAG()
rag.create_index('data/')
print('✅ Índice RAG creado exitosamente')
"
```

El proceso creará:
- `data/faiss_indexes/vectorized_db.bin` - Vectores FAISS
- `data/faiss_indexes/vectorized_db_meta.txt` - Metadatos

> **⚠️ Nota**: Se usa FAISS CPU por defecto. Si tienes CUDA, instala `faiss-gpu` para mejor performance.

---

## 💬 Usar el Agente

### Ejecutar el Chat
```bash
python chat_agente.py
```

### Ejemplo de Interacción
```
💬  Escribe 'exit' para terminar.

👤: ¿Qué son las redes neuronales?
🤖: Un cerebro artificial, con conexiones digitales,
    Las redes neuronales son sistemas computacionales
    Que imitan el funcionamiento del cerebro humano.
    
    Son como una red de neuronas interconectadas,
    Cada una procesando información y pasándola
    A sus vecinas, creando patrones de aprendizaje
    Que permiten resolver problemas complejos.

👤: ¿Cómo funciona el deep learning?
🤖: Capas de aprendizaje, cada vez más profundas,
    El deep learning usa múltiples capas neuronales
    Para extraer características cada vez más abstractas.
    
    Es como un artista que pinta en capas:
    Primero los trazos básicos, luego los detalles,
    Y finalmente la obra maestra emerge
    De la combinación de todas las capas.

👤: exit
👋 ¡Hasta luego!
```

---

## ⚙️ Personalización

### Cambios Rápidos

| Qué cambiar | Dónde | Ejemplo |
|-------------|-------|---------|
| **Modelo LLM** | `src/config/config.py` → `get_chat_model()` | `gpt-4o-2024-08-06` |
| **Prompt del sistema** | `src/config/prompt.py` | Cambiar tono, idioma, comportamiento |
| **Nuevas herramientas** | `src/tools/` + `src/components/agent_builder.py` | Calculadora, web search, etc. |
| **Configuración RAG** | `src/tools/rag.py` | `chunk_size`, `overlap` |
| **Nivel de logging** | Variable `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` |

### Configuración Avanzada

#### Variables de Entorno Disponibles
```bash
# Obligatorio
OPENAI_API_KEY=sk-your-key-here

# Modelo y parámetros
MODEL_NAME=gpt-4o-mini
MODEL_TEMPERATURE=0
MODEL_MAX_TOKENS=1024

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true

# RAG
RAG_CHUNK_SIZE=1000
RAG_OVERLAP=25
RAG_EMBEDDING_MODEL=text-embedding-3-large

# LangSmith (opcional)
LANGCHAIN_TRACING_V2=false
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=mi_agente
```

#### Niveles de Logging
- `DEBUG`: Información detallada para desarrollo
- `INFO`: Información general (por defecto)
- `WARNING`: Solo advertencias y errores
- `ERROR`: Solo errores críticos

---

## 🔧 Desarrollo

### Añadir Nuevas Herramientas

1. **Crear el archivo de herramienta** en `src/tools/`:
```python
from langchain.tools import tool

@tool
def mi_nueva_herramienta(parametro: str) -> str:
    """Descripción de lo que hace la herramienta."""
    # Implementación aquí
    return resultado
```

2. **Añadir al agente** en `src/components/agent_builder.py`:
```python
from src.tools.mi_nueva_herramienta import mi_nueva_herramienta

tools = [Herramienta_RAG, mi_nueva_herramienta]
```

3. **Documentar** su uso en el README

### Estructura de Desarrollo
```
src/
├── components/     # Componentes del agente
├── config/        # Configuración y prompts
├── tools/         # Herramientas disponibles
└── utils/         # Utilidades comunes
```

### Logs y Debugging
- Los logs se guardan en `logs/mi_agente_YYYYMMDD_HHMMSS.log`
- Usa `LOG_LEVEL=DEBUG` para información detallada
- Revisa los logs para debugging y monitoreo

---

## 📊 Monitoreo y Métricas

### Logs Automáticos
- **Sesiones**: Cada ejecución genera un archivo de log único
- **Búsquedas RAG**: Se registran todas las consultas semánticas
- **Errores**: Captura y registra errores automáticamente
- **Performance**: Tiempo de respuesta y uso de tokens

### Archivos de Log
```
logs/
├── mi_agente_20240115_143022.log  # Sesión específica
├── mi_agente_20240115_150145.log  # Otra sesión
└── ...
```

### Información Registrada
- Timestamp de cada interacción
- Consultas RAG realizadas
- Errores y excepciones
- Tiempo de respuesta del LLM
- Uso de tokens

---

## 🛠️ Solución de Problemas

### Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `"Faiss GPU ... not defined"` | No tienes FAISS-GPU instalado | Ignóralo si usas CPU, o instala `faiss-gpu` |
| `"OPENAI_API_KEY not set"` | Variable de entorno faltante | Crea archivo `.env` con tu API key |
| `"Se repite el texto del usuario"` | Problema en el streaming | Asegúrate de usar el último bloque `for ev in agente.stream()` |
| `"Consume demasiados tokens"` | Respuestas muy largas | Reduce `max_tokens` o ajusta el prompt |
| `"No se encuentra el índice RAG"` | Índice no creado | Ejecuta `prueba.ipynb` para crear el índice |
| `"Error de conexión a OpenAI"` | Problemas de red/API | Verifica tu conexión y API key |

### Debugging Avanzado

1. **Activar logging detallado**:
```bash
export LOG_LEVEL=DEBUG
python chat_agente.py
```

2. **Verificar configuración**:
```python
from src.config.config import get_chat_model
model = get_chat_model()
print(f"Modelo configurado: {model.model_name}")
```

3. **Probar RAG directamente**:
```python
from src.tools.rag import RAG
rag = RAG()
results = rag.query("tu consulta", k=3)
print(results)
```

---

## 🤝 Contribuir

### Cómo Contribuir

1. **Fork** el repositorio
2. **Crea** una rama para tu feature:
   ```bash
   git checkout -b feature/nueva-herramienta
   ```
3. **Commit** tus cambios:
   ```bash
   git commit -am 'Añadir nueva herramienta de búsqueda web'
   ```
4. **Push** a la rama:
   ```bash
   git push origin feature/nueva-herramienta
   ```
5. **Crea** un Pull Request

### Estándares de Código

- ✅ Usa **type hints** en todas las funciones
- ✅ Añade **docstrings** descriptivos
- ✅ Sigue el **formato de logging** establecido
- ✅ Incluye **tests** para nuevas funcionalidades
- ✅ Mantén la **arquitectura modular**
- ✅ Documenta **nuevas herramientas**

### Áreas de Mejora

- 🔍 **Más herramientas**: Web search, calculadora, etc.
- 🎨 **Interfaz web**: Dashboard para monitoreo
- 📊 **Métricas avanzadas**: Análisis de uso y performance
- 🔐 **Autenticación**: Sistema de usuarios
- 🌐 **API REST**: Endpoints para integración

---

## 📝 Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- [LangChain](https://github.com/langchain-ai/langchain) por el framework
- [LangGraph](https://github.com/langchain-ai/langgraph) por la orquestación
- [FAISS](https://github.com/facebookresearch/faiss) por la búsqueda vectorial
- [OpenAI](https://openai.com/) por los modelos de lenguaje

---

**⭐ Si este proyecto te es útil, ¡dale una estrella en GitHub!**
