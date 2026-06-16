# ✅ Checklist de la entrega

Stack: **LangChain + LangGraph + Google Gemini** (como el material del curso).

## Requisitos mínimos

- [x] **Conexión con LLM comercial** vía `init_chat_model` — `src/config.py`, `src/assistant.py`
- [x] **Clave en variable de entorno** (`.env`, nunca en el código) — `src/config.py`, `.env.example`
- [x] **Parámetros del modelo expuestos** (temperatura, `max_tokens`) — `src/config.py`
- [x] **RAG: chunking + embeddings** (`RecursiveCharacterTextSplitter` + `GoogleGenerativeAIEmbeddings`) — `src/ingest.py`, `src/indexing.py`
- [x] **Vector store** FAISS persistido en carpeta (`save_local`/`load_local`) — `src/indexing.py`, `storage/faiss/`
- [x] **Respuestas con cita de la fuente** (`[Fuente N: archivo, página, chunk]`) — `src/rag.py`
- [x] **Diálogo multiturno** con memoria (`MemorySaver` + `thread_id`) — `src/graph.py`
- [x] **`get_weather(fecha)` con gestión de errores** (Pydantic + Open-Meteo + fallback) — `src/tools/weather.py`
- [x] **≥3 invocaciones correctas + manejo de fallos** — `notebook.ipynb` §6
- [x] **Todo en un notebook reproducible** — `notebook.ipynb`

## Evaluación y análisis (15%)

- [x] **Conjunto reproducible de prompts** (9 casos) — `notebook.ipynb` §7
- [x] **Métricas** (recuperación, citación, corrección, latencia) — `notebook.ipynb` §7
- [x] **Gráficos** (`matplotlib`) — `notebook.ipynb` §7
- [x] **Casos límite y limitaciones** — `notebook.ipynb` §8, `INFORME.md`

## Calidad de código y documentación (20%)

- [x] **PEP 8 + docstrings + type hints** — `src/`
- [x] **Código modular y reutilizable** — `src/` (7 módulos)
- [x] **Instrucciones de ejecución** — `README.md`
- [x] **`requirements.txt` y `.gitignore`** — raíz
- [x] **Informe** — `INFORME.md`

## Experiencia de usuario (10%)

- [x] **Citas claras** — formato `[Fuente N]`
- [x] **Mensajes de error útiles** (clave ausente, PDF escaneado, fecha inválida)
- [x] **Ejemplos que guían** — `welcome()`, tablas del notebook

## Bonus (+20%)

- [x] **Despliegue web** — `app/streamlit_app.py` (Streamlit)
- [x] **Streaming** en la app
- [x] **Doble proveedor** (Google / OpenAI) — `LLM_PROVIDER`
- [x] **Observabilidad** — `logs/weather.log`, `logs/turns.log`, panel en la app

## Antes de entregar

- [ ] Colocar `TENERIFE.pdf` en `data/`
- [ ] `cp .env.example .env` y poner tu clave
- [ ] `pip install -r requirements.txt`
- [ ] Ejecutar el notebook de arriba abajo
- [ ] **Clear All Outputs** antes de subir a un repositorio
- [ ] **Revocar/regenerar** cualquier clave que se haya podido compartir
