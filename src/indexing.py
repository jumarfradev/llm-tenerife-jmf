"""Embeddings y vector store **FAISS** con LangChain (persistido en carpeta).

* Crea el objeto de embeddings del proveedor activo
  (:class:`GoogleGenerativeAIEmbeddings` u :class:`OpenAIEmbeddings`).
* Construye el índice con :meth:`FAISS.from_documents` y lo **persiste en
  disco** con :meth:`FAISS.save_local`, que crea una **carpeta** con
  ``index.faiss`` (los vectores) e ``index.pkl`` (los textos y metadatos),
  igual que el material del curso.
* En ejecuciones posteriores lo recarga con :meth:`FAISS.load_local` para no
  recalcular embeddings.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import MODEL


def build_embeddings(api_key: str) -> Embeddings:
    """Crea el objeto de embeddings del proveedor activo.

    Args:
        api_key: Clave de la API del proveedor.

    Returns:
        Una implementación de :class:`~langchain_core.embeddings.Embeddings`.
    """
    if MODEL.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=MODEL.embedding_model, api_key=api_key)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=MODEL.embedding_model, google_api_key=api_key)


def build_or_load_index(
    splits: list[Document],
    embeddings: Embeddings,
    faiss_dir: Path,
    force_rebuild: bool = False,
) -> FAISS:
    """Carga el índice FAISS si existe; si no, lo construye y lo persiste.

    Args:
        splits: Fragmentos a indexar (solo se usan si hay que construir).
        embeddings: Objeto de embeddings.
        faiss_dir: Carpeta donde se guarda/lee el índice FAISS.
        force_rebuild: Si ``True``, reconstruye aunque exista la carpeta.

    Returns:
        El vector store :class:`~langchain_community.vectorstores.FAISS`.
    """
    faiss_dir = Path(faiss_dir)
    index_file = faiss_dir / "index.faiss"
    if not force_rebuild and index_file.exists():
        # allow_dangerous_deserialization: el índice lo generamos nosotros, es de confianza.
        return FAISS.load_local(
            str(faiss_dir), embeddings, allow_dangerous_deserialization=True
        )
    vector_store = FAISS.from_documents(splits, embeddings)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(faiss_dir))
    return vector_store
