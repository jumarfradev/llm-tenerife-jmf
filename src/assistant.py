"""Orquestador del asistente turístico de Tenerife (sobre LangChain/LangGraph).

:class:`TenerifeAssistant` construye (o carga) el índice FAISS, crea el modelo
con ``init_chat_model`` y compila el grafo de :mod:`src.graph`. Expone un
método :meth:`chat` que devuelve la respuesta junto con observabilidad
(fuentes citadas, latencia y llamadas a herramientas). La memoria multiturno la
gestiona el *checkpointer* del grafo a través de un ``thread_id``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from .config import MEMORY, MODEL, PATHS, RAG, get_api_key
from .indexing import build_embeddings, build_or_load_index
from .ingest import build_splits_from_pdf
from .graph import build_graph
from .tools.weather import build_weather_tool

# Logging estructurado de turnos (observabilidad).
PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
turn_logger = logging.getLogger("assistant.turns")
if not turn_logger.handlers:
    turn_logger.setLevel(logging.INFO)
    _h = logging.FileHandler(PATHS.turns_log_path, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    turn_logger.addHandler(_h)

WELCOME_MESSAGE = """\
¡Hola! 🌋 Soy tu asistente turístico de Tenerife. Respondo a partir de una \
guía de la isla y siempre te cito de dónde saco la información.

Puedes preguntarme cosas como:
  • ¿Qué playa de arena negra recomiendas para ver el atardecer?
  • ¿Dónde puedo comer en la zona norte?
  • ¿Por qué carretera se sube al Teide?
  • Recomiéndame un guachinche.
  • ¿Qué tiempo hará el sábado para subir al Teide?

Si algo no está en la guía, te lo diré en lugar de inventarlo.
"""


@dataclass
class ToolCall:
    """Registro de una invocación de herramienta dentro de un turno."""

    name: str
    arguments: dict
    result: str


@dataclass
class AssistantTurn:
    """Resultado de un turno (respuesta + observabilidad)."""

    answer: str
    citations: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    sources: list[dict] = field(default_factory=list)


def _content_to_text(content) -> str:
    """Normaliza el contenido de un mensaje a texto plano.

    Algunos modelos (p. ej. Gemini 2.5 con razonamiento) devuelven el contenido
    como una lista de bloques (``[{"type": "text", "text": "..."}]``) en lugar de
    una cadena. Esta función extrae y concatena el texto, ignorando bloques que
    no son de texto (firmas/razonamiento).

    Args:
        content: El ``content`` de un mensaje de LangChain (str o lista).

    Returns:
        El texto de la respuesta como una sola cadena.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = [
            bloque["text"] if isinstance(bloque, dict) and isinstance(bloque.get("text"), str)
            else bloque if isinstance(bloque, str) else ""
            for bloque in content
        ]
        texto = "".join(partes).strip()
        return texto or str(content)
    return str(content)


class TenerifeAssistant:
    """Asistente conversacional con RAG, memoria y herramientas (LangGraph)."""

    def __init__(self, graph, vector_store, use_real_weather: bool = True) -> None:
        self.graph = graph
        self.vector_store = vector_store
        self.use_real_weather = use_real_weather
        self.thread_id = str(uuid.uuid4())  # identifica la conversación (memoria)

    @classmethod
    def from_config(cls, force_rebuild: bool = False, use_real_weather: bool = True) -> "TenerifeAssistant":
        """Crea el asistente leyendo la configuración y el índice persistido.

        Args:
            force_rebuild: Fuerza reconstruir el índice FAISS.
            use_real_weather: Si ``False``, ``get_weather`` usa el modo simulado.

        Returns:
            Un :class:`TenerifeAssistant` listo para conversar.
        """
        PATHS.ensure()
        api_key = get_api_key(MODEL.provider)
        embeddings = build_embeddings(api_key)
        splits = build_splits_from_pdf(
            PATHS.pdf_path, chunk_size=RAG.chunk_size, chunk_overlap=RAG.chunk_overlap
        )
        vector_store = build_or_load_index(
            splits, embeddings, faiss_dir=PATHS.faiss_dir, force_rebuild=force_rebuild
        )
        llm = init_chat_model(
            MODEL.generation_model,
            model_provider=MODEL.model_provider,
            temperature=MODEL.temperature,
            top_p=MODEL.top_p,
            max_tokens=MODEL.max_output_tokens,
        )
        weather_tool = build_weather_tool(use_real_api=use_real_weather)
        graph = build_graph(
            llm, vector_store, weather_tool,
            top_k=RAG.top_k, max_history_tokens=MEMORY.max_history_tokens,
        )
        return cls(graph, vector_store, use_real_weather=use_real_weather)

    def chat(self, message: str) -> AssistantTurn:
        """Procesa un mensaje del usuario y devuelve la respuesta + metadatos.

        Args:
            message: Mensaje del usuario.

        Returns:
            Un :class:`AssistantTurn`.
        """
        start = time.perf_counter()
        config = {"configurable": {"thread_id": self.thread_id}}
        result = self.graph.invoke({"messages": [HumanMessage(message)]}, config=config)

        messages = result["messages"]
        answer = _content_to_text(messages[-1].content)
        sources = result.get("sources", [])
        citations = [
            f"{s.get('source')}, pág. {s.get('page')}, chunk {s.get('chunk_id')}" for s in sources
        ]
        tool_calls = self._extract_tool_calls(messages, message)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        turn = AssistantTurn(
            answer=answer, citations=citations, tool_calls=tool_calls,
            latency_ms=latency_ms, sources=sources,
        )
        self._log_turn(message, turn)
        return turn

    @staticmethod
    def _extract_tool_calls(messages, current_message: str) -> list[ToolCall]:
        """Extrae las llamadas a herramientas del turno actual (observabilidad)."""
        # Localiza el inicio del turno (último mensaje humano con este contenido).
        start_idx = 0
        for i, m in enumerate(messages):
            if getattr(m, "type", "") == "human" and m.content == current_message:
                start_idx = i
        calls: list[ToolCall] = []
        pending: dict[str, ToolCall] = {}
        for m in messages[start_idx:]:
            for tc in getattr(m, "tool_calls", None) or []:
                call = ToolCall(name=tc.get("name", "?"), arguments=tc.get("args", {}), result="")
                calls.append(call)
                pending[tc.get("id", "")] = call
            if getattr(m, "type", "") == "tool":
                pending.get(getattr(m, "tool_call_id", ""), ToolCall("", {}, "")).result = m.content
        return calls

    def _log_turn(self, message: str, turn: AssistantTurn) -> None:
        """Registra el turno en JSON (observabilidad)."""
        turn_logger.info(
            json.dumps(
                {
                    "pregunta": message,
                    "latencia_ms": turn.latency_ms,
                    "fuentes": turn.citations,
                    "herramientas": [tc.name for tc in turn.tool_calls],
                },
                ensure_ascii=False,
            )
        )

    def reset(self) -> None:
        """Reinicia la memoria de la conversación (nuevo ``thread_id``)."""
        self.thread_id = str(uuid.uuid4())

    @staticmethod
    def welcome() -> str:
        """Mensaje de bienvenida con ejemplos de preguntas."""
        return WELCOME_MESSAGE
