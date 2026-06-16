"""Grafo de LangGraph que orquesta RAG + memoria + function calling.

Sigue el patrón de la sesión de RAG del curso (un
:class:`~langgraph.graph.StateGraph` con nodos ``retrieve`` y ``generate``) y
lo amplía para el asistente final:

* **RAG siempre activo:** ``retrieve`` recupera contexto de la guía en cada
  turno (garantiza respuesta fundamentada y citas).
* **Function calling:** ``generate`` ofrece la herramienta ``get_weather`` al
  modelo (``bind_tools``); si el modelo la pide, un nodo ``tools`` la ejecuta y
  se vuelve a ``generate`` para integrar el resultado.
* **Memoria multiturno:** se compila con un *checkpointer* ``MemorySaver``
  (memoria por ``thread_id``) y se recorta el historial con ``trim_messages``
  para no superar el presupuesto de tokens.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .rag import SYSTEM_PROMPT, format_docs_with_sources, sources_from_docs


class AssistantState(MessagesState):
    """Estado del grafo: historial de mensajes + contexto y fuentes del turno."""

    context: str
    sources: list


def _approx_tokens(messages) -> int:
    """Contador de tokens aproximado y offline (≈ caracteres/4)."""
    return sum(len(getattr(m, "content", "") or "") for m in messages) // 4 + len(messages) * 4


def build_graph(llm, vector_store, weather_tool, top_k: int = 4, max_history_tokens: int = 2000):
    """Construye y compila el grafo del asistente.

    Args:
        llm: Modelo de chat de LangChain (de ``init_chat_model``).
        vector_store: Vector store FAISS para la recuperación.
        weather_tool: Herramienta ``get_weather`` de LangChain.
        top_k: Número de fragmentos a recuperar por turno.
        max_history_tokens: Presupuesto de tokens del historial.

    Returns:
        El grafo compilado (con *checkpointer* de memoria).
    """
    llm_with_tools = llm.bind_tools([weather_tool])

    def retrieve(state: AssistantState) -> dict:
        """Recupera contexto de la guía a partir de la última pregunta del usuario."""
        question = ""
        for message in reversed(state["messages"]):
            if getattr(message, "type", "") == "human":
                question = message.content
                break
        docs = vector_store.similarity_search(question, k=top_k)
        return {"context": format_docs_with_sources(docs), "sources": sources_from_docs(docs)}

    def generate(state: AssistantState) -> dict:
        """Genera la respuesta del modelo con el contexto y el historial."""
        history = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=_approx_tokens,
            max_tokens=max_history_tokens,
            include_system=False,
            start_on="human",
            allow_partial=False,
        )
        system = SystemMessage(
            SYSTEM_PROMPT + "\n\nContexto recuperado:\n" + state.get("context", "")
        )
        response = llm_with_tools.invoke([system] + history)
        return {"messages": [response]}

    builder = StateGraph(AssistantState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("tools", ToolNode([weather_tool]))
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    # generate -> "tools" si el modelo pide la herramienta; si no, -> END.
    builder.add_conditional_edges("generate", tools_condition)
    builder.add_edge("tools", "generate")

    return builder.compile(checkpointer=MemorySaver())
