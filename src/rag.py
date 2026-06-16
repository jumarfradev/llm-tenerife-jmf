"""Prompt y formateo del contexto recuperado (estilo del material del curso).

* :data:`SYSTEM_PROMPT`: comportamiento del asistente (responder solo con la
  guía, admitir si no está, citar la fuente, usar ``get_weather`` para el
  tiempo).
* :func:`format_docs_with_sources`: arma el contexto con fuentes numeradas y
  citables (``[Fuente i: archivo, página X, chunk N]``), igual que el curso.
* :data:`RAG_PROMPT`: :class:`~langchain_core.prompts.ChatPromptTemplate` con
  ``{question}`` y ``{context}``.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "Eres un asistente turístico experto en la isla de Tenerife. Respondes en "
    "español, con tono cercano y útil.\n"
    "Usa ÚNICAMENTE el contexto proporcionado, que procede de una guía "
    "turística de Tenerife. Si el contexto no contiene la respuesta, dilo "
    "claramente (por ejemplo: 'No tengo esa información en la guía de "
    "Tenerife'); NO inventes lugares, precios ni horarios. "
    "Cita SIEMPRE las fuentes que uses al final, con el formato (Fuente N). "
    "Si te preguntan por el tiempo o la previsión para una fecha y lugar, usa "
    "la herramienta get_weather; no inventes el tiempo."
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "Pregunta:\n{question}\n\nContexto recuperado:\n{context}\n\nRespuesta:"),
    ]
)


def format_docs_with_sources(documentos: list[Document]) -> str:
    """Construye el bloque de contexto con fuentes numeradas y citables.

    Args:
        documentos: Fragmentos recuperados del vector store.

    Returns:
        Texto del contexto. Si no hay documentos, un aviso explícito para que
        el modelo admita que no hay información.
    """
    if not documentos:
        return "(No se han encontrado fragmentos relevantes en la guía de Tenerife.)"
    bloques = []
    for i, doc in enumerate(documentos, start=1):
        source = doc.metadata.get("source_name", "guía de Tenerife")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        bloques.append(
            f"[Fuente {i}: {source}, página {page}, chunk {chunk_id}]\n{doc.page_content}"
        )
    return "\n\n".join(bloques)


def sources_from_docs(documentos: list[Document]) -> list[dict]:
    """Extrae los metadatos de fuente de los documentos (para observabilidad)."""
    return [
        {
            "source": doc.metadata.get("source_name"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
        }
        for doc in documentos
    ]
