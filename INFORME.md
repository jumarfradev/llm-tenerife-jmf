# Informe técnico — Asistente turístico de Tenerife

## 1. Objetivo y alcance

Prototipo conversacional reproducible que combina **RAG**, **memoria
multiturno** y **function calling** sobre una guía turística de Tenerife. La
arquitectura sigue el stack del material del curso: **LangChain + LangGraph**
con **Google Gemini**.

## 2. Arquitectura

```
Usuario → Grafo LangGraph
            ├─ retrieve  → FAISS.similarity_search (contexto + fuentes)
            ├─ generate  → init_chat_model + get_weather (bind_tools)
            └─ tools     → ejecuta get_weather y vuelve a generate
          (memoria por thread_id vía MemorySaver)
```

Cada módulo de `src/` tiene una responsabilidad única (config, ingesta,
indexado, prompt/citas, herramientas, grafo y orquestador), lo que facilita el
mantenimiento y las pruebas.

## 3. Decisiones técnicas

### 3.1 Conexión con el LLM
- **`init_chat_model(modelo, model_provider=...)`** de LangChain, igual que el
  curso. Por defecto `model_provider="google_genai"` y
  `gemini-2.5-flash` (plan gratuito; el curso usa `gemini-2.5-flash-lite`,
  intercambiable). La clave se lee de una **variable de entorno** (`.env`),
  nunca del código.
- **Doble proveedor:** con `LLM_PROVIDER=openai` se usa `gpt-4o` +
  `text-embedding-3-small`. Centralizado en `src/config.py`.
- Parámetros expuestos: `temperature=0.2` (reproducibilidad), `top_p=0.95` y
  `max_tokens=800`.

### 3.2 RAG
- **Carga:** `PyPDFLoader` (un documento por página, con metadatos de página).
- **Chunking:** `RecursiveCharacterTextSplitter(chunk_size=500,
  chunk_overlap=80, add_start_index=True)`, replicando los valores del curso. A
  cada fragmento se le añade `chunk_id` y `source_name` para poder citarlo.
- **Embeddings:** `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")`.
- **Vector store:** **FAISS** vía LangChain. Se construye con
  `FAISS.from_documents` y se **persiste en una carpeta** con `save_local`
  (`storage/faiss/` → `index.faiss` con los vectores e `index.pkl` con textos y
  metadatos). En ejecuciones posteriores se recarga con `load_local`, evitando
  recalcular embeddings.
- **Citas:** `format_docs_with_sources` numera las fuentes como
  `[Fuente N: archivo, página X, chunk N]` y el *system prompt* obliga a citar y
  a **admitir** cuando la guía no contiene la respuesta (anti-alucinación).

### 3.3 Diálogo multiturno
- Implementado con un **`StateGraph` de LangGraph** (nodos `retrieve` y
  `generate`, como en la sesión de RAG) ampliado con un nodo `tools`.
- La **memoria** la aporta el *checkpointer* `MemorySaver`: el grafo se invoca
  con un `thread_id` por conversación y mantiene el historial entre turnos. Las
  referencias anafóricas ("¿y por allí?") se resuelven gracias a ese historial.
- El historial se recorta con **`trim_messages`** (`strategy="last"`,
  `start_on="human"`) para no exceder el presupuesto de tokens sin romper los
  pares mensaje-herramienta.

### 3.4 Function calling
- `get_weather(fecha, lugar)` se define como **herramienta de LangChain**
  (`@tool`) y se enlaza al modelo con `bind_tools`. Cuando el modelo la solicita,
  el nodo `tools` (`ToolNode`) la ejecuta y el resultado vuelve a `generate`.
- **Validación con Pydantic** (`WeatherInput`): formato ISO de fecha, no fechas
  pasadas y rango máximo de 16 días; lugar no vacío.
- **Ejecución real** contra **Open-Meteo** (geocodificación + previsión diaria,
  zona horaria `Atlantic/Canary`), con **fallback simulado determinista** ante
  fallos de red o *timeout*, de modo que la herramienta **siempre** devuelve un
  resultado útil en lugar de lanzar una excepción.
- Cada invocación se registra en `logs/weather.log`.

### 3.5 Observabilidad
- `TenerifeAssistant.chat` devuelve, además de la respuesta, las **fuentes
  citadas**, las **llamadas a herramientas** (con argumentos y resultado) y la
  **latencia**. Los turnos se registran en `logs/turns.log`.

## 4. Evaluación

Batería fija de 9 prompts (7 de RAG, 1 fuera de corpus, 1 de tiempo). Se mide:

- **Recuperación (hit):** si los fragmentos recuperados contienen los términos
  clave esperados.
- **Citación:** si la respuesta incluye fuentes.
- **Corrección:** términos clave presentes (RAG), admisión de desconocimiento
  (fuera de corpus) o invocación de `get_weather` (tiempo).
- **Latencia** por turno.

Los resultados se muestran en una tabla (`pandas`) y un gráfico de barras
(`matplotlib`). La evaluación reinicia la memoria por prompt y usa temperatura
baja, por lo que es reproducible.

## 5. Seguridad y buenas prácticas

- La clave de API vive solo en `.env` (en `.gitignore`); el repositorio incluye
  únicamente `.env.example`.
- El código sigue PEP 8, con *docstrings* y *type hints*.
- `FAISS.load_local` usa `allow_dangerous_deserialization=True` porque el índice
  lo genera la propia aplicación (origen de confianza).

## 6. Limitaciones y mejoras futuras

- El conocimiento se limita a la guía; no hay precios ni horarios salvo los que
  el documento incluya.
- El *chunking* por caracteres puede cortar frases; el solapamiento lo mitiga.
- El recuento de tokens para recortar el historial es aproximado.
- **Mejoras:** *re-ranking*, recuperación híbrida (densa + BM25), más
  herramientas (transporte, eventos), multimodalidad y evaluación de fidelidad
  asistida por LLM.
