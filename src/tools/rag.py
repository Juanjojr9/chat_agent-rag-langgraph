# rag_local.py
import os, faiss, numpy as np, fitz, docx , logging
from dotenv import load_dotenv
import openai   # SDK v1.13+
from functools import lru_cache
from typing import List, Dict, Optional

rag_local = None
logger = logging.getLogger(__name__)
load_dotenv()   # lee .env (OPENAI_API_KEY, etc.)

class RAGLocal:
    """
    RAG sobre documentos en disco.
    - root_folder: carpeta donde buscar PDF / DOCX / TXT.
    - index_folder: carpeta para guardar índice FAISS y metadatos.
    """
    def __init__(self, root_folder: str, index_folder: str = "faiss_indexes",
                 client: openai.OpenAI | None = None,
                 chunk_size: int = 1000, overlap: int = 25):

        self.root_folder = os.path.abspath(root_folder)
        if not os.path.isdir(self.root_folder):
            raise ValueError(f"No existe la carpeta {self.root_folder}")

        self.index_folder = os.path.abspath(index_folder)
        os.makedirs(self.index_folder, exist_ok=True)

        self.index_path = os.path.join(self.index_folder, "vectorized_db.bin")
        self.meta_path  = os.path.join(self.index_folder, "vectorized_db_meta.txt")

        # Validar parámetros
        if chunk_size < 100:
            raise ValueError("chunk_size debe ser al menos 100")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap debe estar entre 0 y chunk_size")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Inicializar cliente OpenAI
        try:
            self.client = client or openai.OpenAI()
            logger.info("Cliente OpenAI inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar cliente OpenAI: {str(e)}")
            raise
            
        self._docs, self._index, self.dimension = [], None, None

    # ---------- extracción de texto ----------
    @staticmethod
    def _extract_pdf(path: str) -> str:
        """Extrae texto de un archivo PDF."""
        try:
            with fitz.open(path) as doc:
                text = "\n".join(page.get_text() for page in doc)
                logger.debug(f"PDF extraído: {path} ({len(text)} caracteres)")
                return text
        except Exception as e:
            logger.error(f"Error extrayendo PDF {path}: {str(e)}")
            raise

    @staticmethod
    def _extract_docx(path: str) -> str:
        """Extrae texto de un archivo DOCX."""
        try:
            text = "\n".join(p.text for p in docx.Document(path).paragraphs)
            logger.debug(f"DOCX extraído: {path} ({len(text)} caracteres)")
            return text
        except Exception as e:
            logger.error(f"Error extrayendo DOCX {path}: {str(e)}")
            raise

    @staticmethod
    def _extract_txt(path: str) -> str:
        """Extrae texto de un archivo TXT con múltiples encodings."""
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for enc in encodings:
            try:
                with open(path, encoding=enc) as f:
                    text = f.read()
                    logger.debug(f"TXT extraído: {path} ({len(text)} caracteres) con encoding {enc}")
                    return text
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error extrayendo TXT {path} con encoding {enc}: {str(e)}")
                continue
                
        # Último intento con manejo de errores
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
                logger.warning(f"TXT extraído con reemplazo de caracteres: {path}")
                return text
        except Exception as e:
            logger.error(f"Error fatal extrayendo TXT {path}: {str(e)}")
            raise

    # ---------- utilidades ----------
    def _chunk(self, text: str) -> List[str]:
        """Divide el texto en chunks con overlap."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():  # Solo añadir chunks no vacíos
                chunks.append(chunk)
        return chunks

    def _embed(self, text: str) -> np.ndarray:
        """Genera embeddings para un texto."""
        try:
            resp = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=[text]
            )
            return np.asarray(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generando embedding: {str(e)}")
            raise

    # ---------- construcción / carga de índice ----------
    def create_index(self) -> None:
        """Crea el índice FAISS desde los documentos."""
        file_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.root_folder)
            for f in files
            if f.lower().endswith((".pdf", ".docx", ".txt"))
        ]

        if not file_paths:
            raise RuntimeError(f"No se encontraron documentos válidos en {self.root_folder}")

        logger.info(f"Procesando {len(file_paths)} documentos...")
        documents = []
        
        for path in file_paths:
            try:
                ext = path.lower().rsplit(".", 1)[-1]
                if ext == "pdf":
                    text = self._extract_pdf(path)
                elif ext == "docx":
                    text = self._extract_docx(path)
                else:
                    text = self._extract_txt(path)

                # Dividir en chunks
                chunks = self._chunk(text)
                for chunk in chunks:
                    documents.append({"path": path, "text": chunk})
                    
                logger.debug(f"Documento procesado: {path} -> {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error procesando {path}: {str(e)}")
                continue

        if not documents:
            raise RuntimeError("No se pudieron procesar documentos válidos")

        logger.info(f"Generando embeddings para {len(documents)} chunks...")
        
        try:
            # Generar embeddings en lotes
            batch_size = 100  # OpenAI permite hasta 2048 por request
            all_embeds = []
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch = self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=[d["text"] for d in batch_docs],
                )
                batch_embeds = [r.embedding for r in batch.data]
                all_embeds.extend(batch_embeds)
                logger.debug(f"Batch {i//batch_size + 1} procesado: {len(batch_embeds)} embeddings")
            
            embeds = np.vstack(all_embeds)
            self.dimension = embeds.shape[1]
            self._index = faiss.IndexFlatL2(self.dimension)
            self._index.add(embeds)
            self._docs = documents

            # Guardar índice y metadatos
            faiss.write_index(self._index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as fh:
                for d in documents:
                    line = f"{d['path']}|{d['text'].replace(chr(10), ' ')}\n"
                    fh.write(line)

            logger.info(f"Índice creado exitosamente: {len(documents)} documentos, {self.dimension} dimensiones")
            
        except Exception as e:
            logger.error(f"Error creando índice: {str(e)}")
            raise

    def load_index(self) -> None:
        """Carga el índice y metadatos ya existentes."""
        if not (os.path.isfile(self.index_path) and os.path.isfile(self.meta_path)):
            raise FileNotFoundError(f"No existe un índice previo en {self.index_folder}. Ejecuta create_index().")

        try:
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path, encoding="utf-8") as fh:
                self._docs = []
                for line in fh:
                    path, text = line.rstrip("\n").split("|", 1)
                    self._docs.append({"path": path, "text": text})
            self.dimension = self._index.d
            logger.info(f"Índice cargado: {len(self._docs)} documentos, {self.dimension} dimensiones")
            
        except Exception as e:
            logger.error(f"Error cargando índice: {str(e)}")
            raise

    # ---------- consulta ----------
    def query(self, question: str, k: int = 3) -> str:
        """Realiza una consulta al índice RAG."""
        if self._index is None:
            return "Índice no cargado. Usa load_index() o create_index()."

        if not question or not question.strip():
            return "La pregunta no puede estar vacía."

        try:
            q_embed = self._embed(question.strip()).reshape(1, -1)
            dist, idxs = self._index.search(q_embed, k)
            
            answers = []
            for i, idx in enumerate(idxs[0]):
                if idx != -1 and idx < len(self._docs):
                    doc = self._docs[idx]
                    answers.append(f"{i+1}. {doc['text']}\n")
            
            if not answers:
                return "No se encontraron documentos relevantes para tu pregunta."
                
            return "".join(answers)
            
        except Exception as e:
            logger.error(f"Error en consulta RAG: {str(e)}")
            return f"Error interno en la consulta: {str(e)}"
    
# ← objeto global (vacío)

@lru_cache
def init_rag(path: str = "data") -> RAGLocal:
    """
    Carga o crea el índice una sola vez y lo guarda en rag_local.
    Devuelve la instancia para quien quiera usarla.
    """
    global rag_local
    if rag_local is None:
        try:
            rag_local = RAGLocal(root_folder=path, index_folder=path+"/faiss_indexes")
            rag_local.load_index()
            logger.info("RAG inicializado correctamente")
        except FileNotFoundError:
            logger.info("Creando nuevo índice RAG...")
            rag_local.create_index()
        except Exception as e:
            logger.error(f"Error inicializando RAG: {str(e)}")
            raise
    return rag_local
