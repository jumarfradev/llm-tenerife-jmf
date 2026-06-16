"""Carga del PDF y *chunking* con LangChain (igual que el material del curso).

Replica el enfoque de la sesión de RAG del curso:

* Carga el PDF con :class:`~langchain_community.document_loaders.PyPDFLoader`
  (un :class:`~langchain_core.documents.Document` por página, con metadatos de
  página y fuente).
* Trocea con
  :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
  (``chunk_size=500``, ``chunk_overlap=80``, ``add_start_index=True``).
* Enriquece cada fragmento con ``chunk_id`` y ``source_name`` para poder citar
  la fuente (página y nº de chunk).
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAG


def load_pdf(pdf_path: Path) -> list[Document]:
    """Carga el PDF de la guía como una lista de documentos (uno por página).

    Args:
        pdf_path: Ruta al PDF de la guía.

    Returns:
        Lista de :class:`Document`, uno por página.

    Raises:
        FileNotFoundError: Si el PDF no existe.
        ValueError: Si no se pudo extraer texto (PDF escaneado sin OCR).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"No se encontró el PDF de la guía en '{pdf_path}'. "
            f"Coloca TENERIFE.pdf en la carpeta 'data/'."
        )
    docs = PyPDFLoader(str(pdf_path)).load()
    if sum(len(d.page_content) for d in docs) < 200:
        raise ValueError(
            "No se pudo extraer texto del PDF. Si es un escaneado (solo "
            "imágenes), aplícale OCR antes (p. ej. 'ocrmypdf entrada.pdf "
            "TENERIFE.pdf')."
        )
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = RAG.chunk_size,
    chunk_overlap: int = RAG.chunk_overlap,
    source_name: str = "TENERIFE.pdf",
) -> list[Document]:
    """Trocea los documentos en fragmentos con solapamiento.

    Args:
        docs: Documentos de :func:`load_pdf`.
        chunk_size: Tamaño de cada fragmento (caracteres).
        chunk_overlap: Solapamiento entre fragmentos (caracteres).
        source_name: Nombre del documento de origen (para las citas).

    Returns:
        Lista de fragmentos (:class:`Document`) con metadatos ``chunk_id`` y
        ``source_name`` añadidos.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)
    for i, doc in enumerate(splits):
        doc.metadata["chunk_id"] = i
        doc.metadata["source_name"] = source_name
    return splits


def build_splits_from_pdf(
    pdf_path: Path,
    chunk_size: int = RAG.chunk_size,
    chunk_overlap: int = RAG.chunk_overlap,
) -> list[Document]:
    """Atajo: carga el PDF y devuelve los fragmentos listos para indexar.

    Args:
        pdf_path: Ruta al PDF de la guía.
        chunk_size: Tamaño de cada fragmento.
        chunk_overlap: Solapamiento entre fragmentos.

    Returns:
        Lista de fragmentos (:class:`Document`).
    """
    docs = load_pdf(pdf_path)
    return split_documents(
        docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        source_name=Path(pdf_path).name,
    )
