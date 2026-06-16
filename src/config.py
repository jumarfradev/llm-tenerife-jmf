"""Configuración centralizada del asistente turístico de Tenerife.

Concentra los parámetros ajustables (modelos, chunking, rutas) en un único
lugar y carga las variables de entorno desde un ``.env`` con
:mod:`python-dotenv`. La clave de la API se lee del entorno (nunca se escribe
en el código).

El proyecto sigue el stack del material del curso: **LangChain + LangGraph**
con **Google Gemini** (`init_chat_model` + `GoogleGenerativeAIEmbeddings`),
admitiendo también OpenAI cambiando ``LLM_PROVIDER``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class ModelConfig:
    """Parámetros del modelo de lenguaje y de los embeddings.

    Attributes:
        provider: ``"google"`` (por defecto) u ``"openai"``.
        model_provider: Nombre del proveedor para ``init_chat_model`` de
            LangChain (``"google_genai"`` u ``"openai"``).
        generation_model: Modelo generativo (el curso usa
            ``gemini-2.5-flash-lite``; aquí ``gemini-2.5-flash`` por dar algo
            más de calidad, ambos del plan gratuito).
        embedding_model: Modelo de embeddings. Para Gemini se usa el prefijo
            ``models/`` que espera ``GoogleGenerativeAIEmbeddings``.
        temperature: Aleatoriedad (baja para reproducibilidad).
        top_p: Muestreo por núcleo (nucleus sampling).
        max_output_tokens: Tope de tokens de salida.
    """

    provider: str = "google"
    model_provider: str = "google_genai"
    generation_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.2
    top_p: float = 0.95
    max_output_tokens: int = 800


@dataclass(frozen=True)
class RAGConfig:
    """Parámetros de la canalización RAG (chunking y recuperación).

    Se replican los valores del material del curso: ``chunk_size=500`` y
    ``chunk_overlap=80`` con ``RecursiveCharacterTextSplitter``.
    """

    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4


@dataclass(frozen=True)
class MemoryConfig:
    """Parámetros de la memoria multiturno (presupuesto de tokens)."""

    max_history_tokens: int = 2000


@dataclass(frozen=True)
class Paths:
    """Rutas del proyecto (datos, índice FAISS persistido, logs)."""

    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    storage_dir: Path = PROJECT_ROOT / "storage"
    logs_dir: Path = PROJECT_ROOT / "logs"
    pdf_path: Path = PROJECT_ROOT / "data" / "TENERIFE.pdf"
    # Carpeta del índice FAISS (LangChain.save_local crea index.faiss + index.pkl).
    faiss_dir: Path = PROJECT_ROOT / "storage" / "faiss"
    weather_log_path: Path = PROJECT_ROOT / "logs" / "weather.log"
    turns_log_path: Path = PROJECT_ROOT / "logs" / "turns.log"

    def ensure(self) -> None:
        """Crea los directorios necesarios si no existen."""
        for directory in (self.data_dir, self.storage_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)


# Valores por defecto por proveedor (modelos y nombre para init_chat_model).
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "google": {
        "model_provider": "google_genai",
        "generation_model": "gemini-2.5-flash",
        "embedding_model": "models/gemini-embedding-001",
    },
    "openai": {
        "model_provider": "openai",
        # gpt-4o es heredado en 2026; cámbialo por un modelo actual si hace falta.
        "generation_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
}

_PROVIDER_ENV_VAR: dict[str, str] = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_model_config() -> ModelConfig:
    """Construye la configuración del modelo según ``LLM_PROVIDER``.

    Returns:
        Una :class:`ModelConfig` para el proveedor activo (``google`` por
        defecto).
    """
    provider = os.getenv("LLM_PROVIDER", "google").strip().lower()
    if provider not in _PROVIDER_DEFAULTS:
        provider = "google"
    return ModelConfig(provider=provider, **_PROVIDER_DEFAULTS[provider])


MODEL = build_model_config()
RAG = RAGConfig()
MEMORY = MemoryConfig()
PATHS = Paths()


def get_api_key(provider: str | None = None) -> str:
    """Devuelve la clave de la API del proveedor activo, leída del entorno.

    Args:
        provider: ``"google"`` u ``"openai"``. Si es ``None``, usa el de
            :data:`MODEL`.

    Returns:
        La clave de la API.

    Raises:
        RuntimeError: Si la variable de entorno no está definida.
    """
    provider = (provider or MODEL.provider).strip().lower()
    env_var = _PROVIDER_ENV_VAR.get(provider, "GOOGLE_API_KEY")
    key = os.getenv(env_var)
    if not key:
        url = (
            "https://aistudio.google.com/apikey"
            if provider == "google"
            else "https://platform.openai.com/api-keys"
        )
        raise RuntimeError(
            f"No se encontró la variable de entorno '{env_var}' (proveedor "
            f"'{provider}'). Crea un '.env' a partir de '.env.example' con:\n"
            f"    {env_var}=tu_clave_aqui\nClave en: {url}"
        )
    return key
