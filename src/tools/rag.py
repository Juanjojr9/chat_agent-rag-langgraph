# rag_local.py
import os, faiss, numpy as np, fitz, docx , logging
from dotenv import load_dotenv
import openai   # SDK v1.13+
from functools import lru_cache
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

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.client = client or openai.OpenAI()  # usa OPENAI_API_KEY del entorno
        self._docs, self._index, self.dimension = [], None, None

    # ---------- extracción de texto ----------
    @staticmethod
    def _extract_pdf(path):
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)

    @staticmethod
    def _extract_docx(path):
        return "\n".join(p.text for p in docx.Document(path).paragraphs)

    @staticmethod
    def _extract_txt(path):
        for enc in ("utf‑8", "latin‑1", "cp1252"):
            try:
                with open(path, encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, encoding="utf‑8", errors="replace") as f:
            return f.read()

    # ---------- utilidades ----------
    def _chunk(self, text):
        for i in range(0, len(text), self.chunk_size - self.overlap):
            yield text[i : i + self.chunk_size]

    def _embed(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=[text]
        )
        return np.asarray(resp.data[0].embedding, dtype=np.float32)

    # ---------- construcción / carga de índice ----------
    def create_index(self):
        file_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.root_folder)
            for f in files
            if f.lower().endswith((".pdf", ".docx", ".txt"))
        ]

        if not file_paths:
            raise RuntimeError("No se encontraron documentos válidos.")

        documents = []
        for path in file_paths:
            ext = path.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                text = self._extract_pdf(path)
            elif ext == "docx":
                text = self._extract_docx(path)
            else:
                text = self._extract_txt(path)

            for chunk in self._chunk(text):
                documents.append({"path": path, "text": chunk})

        batch = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=[d["text"] for d in documents],
            )
        embeds = np.vstack([r.embedding for r in batch.data])
        self.dimension = embeds.shape[1]
        self._index = faiss.IndexFlatL2(self.dimension)
        self._index.add(embeds)
        self._docs = documents

        faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "w", encoding="utf‑8") as fh:
            for d in documents:
                line = f"{d['path']}|{d['text'].replace(chr(10), ' ')}\n"
                fh.write(line)

        logger.info("Índice creado en %s; metadatos en %s", self.index_path, self.meta_path)

    def load_index(self):
        """Carga el índice y metadatos ya existentes."""
        if not (os.path.isfile(self.index_path) and os.path.isfile(self.meta_path)):
            raise FileNotFoundError("No existe un índice previo. Ejecuta create_index().")

        self._index = faiss.read_index(self.index_path)
        with open(self.meta_path, encoding="utf‑8") as fh:
            self._docs = []
            for line in fh:
                path, text = line.rstrip("\n").split("|", 1)
                self._docs.append({"path": path, "text": text})
        self.dimension = self._index.d

    # ---------- consulta ----------
    def query(self, question: str, k: int = 3) -> str:
        if self._index is None:
            return "Índice no cargado. Usa load_index() o create_index()."

        q_embed = self._embed(question).reshape(1, -1)
        dist, idxs = self._index.search(q_embed, k)
        answers = [
            f"{i+1}. {self._docs[idx]['text']}\n"
            for i, idx in enumerate(idxs[0])
            if idx != -1
        ]
        return "".join(answers) or "No se encontraron documentos relevantes."
    
                     # ← objeto global (vacío)

@lru_cache
def init_rag(path: str = "data") -> RAGLocal:
    """
    Carga o crea el índice una sola vez y lo guarda en rag_local.
    Devuelve la instancia para quien quiera usarla.
    """
    global rag_local
    if rag_local is None:
        rag_local = RAGLocal(root_folder=path, index_folder=path+"/faiss_indexes")
        try:
            rag_local.load_index()
        except FileNotFoundError:
            rag_local.create_index()
    return rag_local
