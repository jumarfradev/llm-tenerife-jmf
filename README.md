# 🌋 Asistente turístico conversacional de Tenerife

Asistente que combina **RAG** sobre una guía turística de Tenerife, **memoria
multiturno** y **llamadas a funciones externas** (`get_weather`). Proyecto final
de la asignatura *Large Language Models*.

Construido con el **mismo stack del material del curso**: **LangChain +
LangGraph** con **Google Gemini** (`init_chat_model`,
`GoogleGenerativeAIEmbeddings`, `PyPDFLoader`, `RecursiveCharacterTextSplitter`,
`FAISS`, `StateGraph`). Admite también **OpenAI** cambiando una variable de
entorno.

## ✨ Características

- **RAG con citas.** La guía se trocea con `RecursiveCharacterTextSplitter`
  (`chunk_size=500`, `chunk_overlap=80`), se indexa en **FAISS** y se persiste en
  la carpeta `storage/faiss/` (`index.faiss` + `index.pkl`). Cada respuesta cita
  la fuente (`[Fuente N: archivo, página X, chunk N]`).
- **Memoria multiturno.** Grafo de **LangGraph** compilado con un *checkpointer*
  `MemorySaver`; el historial se recorta con `trim_messages` para no exceder el
  presupuesto de tokens.
- **Function calling.** `get_weather(fecha, lugar)` como herramienta de
  LangChain: previsión real con **Open-Meteo** (sin clave), *fallback* simulado,
  validación con **Pydantic** y manejo de errores con mensajes útiles.
- **Evaluación reproducible.** Batería fija de prompts con métricas (recuperación,
  citación, corrección, latencia) y gráficos.
- **Bonus.** App web con **Streamlit** (streaming + panel de observabilidad),
  doble proveedor (Google/OpenAI) y *logging* de turnos y de la herramienta.

## 📁 Estructura

```
tenerife-rag-assistant/
├── notebook.ipynb            # Notebook principal (orquesta todo)
├── src/
│   ├── config.py             # Configuración (modelos, rutas, proveedor dual)
│   ├── ingest.py             # PyPDFLoader + RecursiveCharacterTextSplitter
│   ├── indexing.py           # Embeddings + FAISS (save_local/load_local)
│   ├── rag.py                # Prompt (ChatPromptTemplate) + formato de citas
│   ├── graph.py              # Grafo LangGraph: retrieve→generate→tools + memoria
│   ├── assistant.py          # TenerifeAssistant (orquestador de alto nivel)
│   └── tools/
│       └── weather.py        # Herramienta @tool get_weather (Open-Meteo)
├── app/
│   └── streamlit_app.py      # App web (bonus)
├── data/                     # TENERIFE.pdf (no se versiona)
├── storage/faiss/            # Índice FAISS persistido (se genera solo)
├── logs/                     # weather.log, turns.log
├── requirements.txt
├── .env.example              # Plantilla de variables de entorno
└── .gitignore
```

## 🚀 Puesta en marcha

```bash
# 1. Dependencias
pip install -r requirements.txt

# 2. Guía turística
#    Copia TENERIFE.pdf en data/

# 3. Clave de API
cp .env.example .env          # y edita .env con tu clave
#    Por defecto usa Google Gemini -> GOOGLE_API_KEY
#    (clave gratuita en https://aistudio.google.com/apikey)

# 4a. Notebook
jupyter notebook notebook.ipynb

# 4b. App web (bonus)
streamlit run app/streamlit_app.py
```

## 🔧 Configuración

Variables de entorno (en `.env`):

| Variable | Descripción | Por defecto |
|----------|-------------|-------------|
| `LLM_PROVIDER` | `google` u `openai` | `google` |
| `GOOGLE_API_KEY` | Clave de Google Gemini | — |
| `OPENAI_API_KEY` | Clave de OpenAI (solo si `LLM_PROVIDER=openai`) | — |

Los parámetros del modelo (temperatura, `max_tokens`), el *chunking* y `top_k`
se ajustan en `src/config.py`.

## 🧪 Cómo cumple la rúbrica

| Requisito | Dónde |
|-----------|-------|
| Conexión LLM con clave en variable de entorno | `src/config.py`, `src/assistant.py` |
| RAG (chunking + embeddings + vector store + citas) | `src/ingest.py`, `src/indexing.py`, `src/rag.py` |
| Diálogo multiturno | `src/graph.py` (`MemorySaver` + `thread_id`) |
| ≥3 *function calls* con manejo de fallos | `src/tools/weather.py`, notebook §6 |
| Evaluación con métricas y gráficos | notebook §7 |
| Calidad de código (PEP8, docstrings, modular) | `src/` |
| UX (citas, errores útiles, ejemplos) | `welcome()`, app Streamlit |
| Bonus (web, doble proveedor, observabilidad) | `app/`, `config.py`, logs |

## ⚠️ Notas

- El índice FAISS depende del modelo de embeddings: si cambias de proveedor,
  reconstruye el índice (`force_rebuild=True` o borra `storage/faiss/`).
- Si tu PDF es un escaneado (solo imágenes), aplícale OCR antes
  (`ocrmypdf entrada.pdf TENERIFE.pdf`).
- Antes de subir el notebook a un repositorio, haz *Clear All Outputs* para no
  filtrar datos ni la clave.
