"""App web de chat (bonus) para el asistente turístico de Tenerife.

Reutiliza los módulos de ``src/`` y añade:

* Interfaz de chat con **streaming** de la respuesta (revelado en tiempo real).
* Panel de **observabilidad** por turno: fuentes citadas, latencia y llamadas a
  herramientas.

Ejecuta con:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Permite importar el paquete src/ al ejecutar desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant import TenerifeAssistant  # noqa: E402
from src.config import MODEL  # noqa: E402

st.set_page_config(page_title="Asistente de Tenerife", page_icon="🌋", layout="centered")


@st.cache_resource(show_spinner="Construyendo el índice y el asistente…")
def cargar_asistente() -> TenerifeAssistant:
    """Crea el asistente una sola vez (cacheado entre interacciones)."""
    return TenerifeAssistant.from_config(use_real_weather=True)


def stream_texto(texto: str):
    """Revela el texto palabra a palabra para simular streaming en la UI.

    Se re-emite la respuesta ya calculada para mantener intacta la orquestación
    de *function calling* del grafo (que es síncrona).
    """
    for palabra in texto.split(" "):
        yield palabra + " "
        time.sleep(0.02)


st.title("🌋 Asistente turístico de Tenerife")
st.caption(
    f"RAG sobre la guía · memoria multiturno · function calling · "
    f"proveedor: **{MODEL.provider}** ({MODEL.generation_model})"
)

# --- Carga del asistente (con manejo de error de clave) --------------------- #
try:
    asistente = cargar_asistente()
except Exception as exc:  # noqa: BLE001
    st.error(
        "No se pudo iniciar el asistente. Comprueba que has creado el archivo "
        "`.env` con tu clave y que `data/TENERIFE.pdf` existe.\n\n"
        f"Detalle: {exc}"
    )
    st.stop()

# --- Estado de la conversación --------------------------------------------- #
if "historial" not in st.session_state:
    st.session_state.historial = []  # lista de dicts: {role, content, meta}
    st.session_state.bienvenida = asistente.welcome()

# --- Barra lateral: ayuda y reinicio ---------------------------------------- #
with st.sidebar:
    st.header("ℹ️ Cómo usarlo")
    st.markdown(st.session_state.bienvenida)
    if st.button("🗑️ Reiniciar conversación"):
        asistente.reset()
        st.session_state.historial = []
        st.rerun()

# --- Render del historial --------------------------------------------------- #
for msg in st.session_state.historial:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            with st.expander("🔎 Observabilidad del turno"):
                if meta["citations"]:
                    st.markdown("**Fuentes citadas:** " + ", ".join(meta["citations"]))
                if meta["tool_calls"]:
                    for tc in meta["tool_calls"]:
                        st.markdown(
                            f"**🔧 {tc['name']}**(`{tc['arguments']}`) → {tc['result']}"
                        )
                st.markdown(f"**Latencia:** {meta['latency_ms']} ms")

# --- Entrada del usuario ---------------------------------------------------- #
if pregunta := st.chat_input("Pregúntame sobre Tenerife…"):
    st.session_state.historial.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            turno = asistente.chat(pregunta)
        # Streaming visible de la respuesta final.
        st.write_stream(stream_texto(turno.answer))

        meta = {
            "citations": turno.citations,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result": tc.result}
                for tc in turno.tool_calls
            ],
            "latency_ms": turno.latency_ms,
        }
        with st.expander("🔎 Observabilidad del turno"):
            if meta["citations"]:
                st.markdown("**Fuentes citadas:** " + ", ".join(meta["citations"]))
            if meta["tool_calls"]:
                for tc in meta["tool_calls"]:
                    st.markdown(
                        f"**🔧 {tc['name']}**(`{tc['arguments']}`) → {tc['result']}"
                    )
            st.markdown(f"**Latencia:** {meta['latency_ms']} ms")

    st.session_state.historial.append(
        {"role": "assistant", "content": turno.answer, "meta": meta}
    )
