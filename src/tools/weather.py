"""Herramienta ``get_weather``: previsión meteorológica para Tenerife.

Definida como **herramienta de LangChain** (`@tool`) para que el modelo la
invoque mediante *tool calling*. Cumple los requisitos del enunciado:

* **Esquema con Pydantic** (:class:`WeatherInput`) con validación de fecha y lugar.
* **Ejecución real** contra **Open-Meteo** (sin clave): geocodificación +
  previsión diaria.
* **Fallback simulado determinista** si no hay red.
* **Manejo de errores** (fecha inválida, lugar no encontrado, timeout/red)
  devolviendo siempre un texto útil.
* **Logging** de cada intento (argumentos, resultado/error y *timestamp*).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

# --- Logging ---------------------------------------------------------------- #
_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "weather.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("tools.weather")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8"); _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler(); _sh.setFormatter(_fmt)
    logger.addHandler(_fh); logger.addHandler(_sh)

MAX_FORECAST_DAYS = 16
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

WMO_CODES: dict[int, str] = {
    0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado",
    3: "nublado", 45: "niebla", 48: "niebla con escarcha", 51: "llovizna ligera",
    53: "llovizna moderada", 55: "llovizna intensa", 61: "lluvia ligera",
    63: "lluvia moderada", 65: "lluvia fuerte", 71: "nevada ligera",
    80: "chubascos ligeros", 81: "chubascos moderados", 82: "chubascos violentos",
    95: "tormenta",
}


class WeatherInput(BaseModel):
    """Argumentos validados de una consulta meteorológica."""

    fecha: str = Field(..., description="Fecha en formato ISO YYYY-MM-DD.")
    lugar: str = Field(..., description="Nombre del lugar en Tenerife (p. ej. 'Teide').")

    @field_validator("fecha")
    @classmethod
    def _validar_fecha(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"La fecha '{value}' no es válida. Usa YYYY-MM-DD.") from exc
        today = date.today()
        if parsed < today:
            raise ValueError(f"La fecha '{value}' ya ha pasado.")
        if parsed > today + timedelta(days=MAX_FORECAST_DAYS):
            raise ValueError(
                f"Solo hay previsión hasta {MAX_FORECAST_DAYS} días vista "
                f"(hasta {today + timedelta(days=MAX_FORECAST_DAYS)})."
            )
        return value.strip()

    @field_validator("lugar")
    @classmethod
    def _validar_lugar(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("El lugar no puede estar vacío.")
        return value.strip()


def _simulated_forecast(fecha: str, lugar: str) -> dict[str, Any]:
    """Previsión simulada determinista (sin red), para demos/evaluación offline."""
    seed = int(hashlib.md5(f"{lugar.lower()}|{fecha}".encode()).hexdigest(), 16)
    temp_min = 14 + (seed % 8)
    temp_max = temp_min + 5 + (seed % 6)
    code = [0, 1, 2, 3, 61, 80][seed % 6]
    return {
        "ok": True, "fuente": "simulada (sin conexión)", "lugar": lugar, "fecha": fecha,
        "temp_min_c": temp_min, "temp_max_c": temp_max,
        "descripcion": WMO_CODES.get(code, "variable"),
    }


def _geocode(lugar: str, timeout: int) -> tuple[float, float, str]:
    """Resuelve un nombre de lugar a coordenadas con Open-Meteo."""
    resp = requests.get(
        OPEN_METEO_GEOCODE, params={"name": lugar, "count": 1, "language": "es"}, timeout=timeout
    )
    resp.raise_for_status()
    results = (resp.json() or {}).get("results") or []
    if not results:
        raise LookupError(f"No se encontró el lugar '{lugar}'.")
    top = results[0]
    return float(top["latitude"]), float(top["longitude"]), top.get("name", lugar)


def get_weather(fecha: str, lugar: str, timeout: int = 10, use_real_api: bool = True) -> dict[str, Any]:
    """Devuelve la previsión meteorológica para una fecha y un lugar.

    Valida los argumentos, consulta Open-Meteo (o usa el *fallback* simulado) y
    registra el intento. Nunca lanza excepciones: devuelve un dict con ``ok``.

    Args:
        fecha: Fecha ISO ``YYYY-MM-DD``.
        lugar: Nombre del lugar en Tenerife.
        timeout: Timeout de red (segundos).
        use_real_api: Si ``False``, fuerza el modo simulado.

    Returns:
        Diccionario con la previsión (``ok=True``) o el error (``ok=False``).
    """
    logger.info("get_weather solicitado | args=%s", {"fecha": fecha, "lugar": lugar})
    try:
        query = WeatherInput(fecha=fecha, lugar=lugar)
    except Exception as exc:
        msg = str(exc.errors()[0]["msg"]) if hasattr(exc, "errors") else str(exc)
        result = {"ok": False, "error": "fecha_o_lugar_invalido", "mensaje": msg}
        logger.warning("get_weather validación | %s", result)
        return result

    if not use_real_api:
        result = _simulated_forecast(query.fecha, query.lugar)
        logger.info("get_weather (simulado) | %s", result)
        return result

    try:
        lat, lon, nombre = _geocode(query.lugar, timeout=timeout)
        fc = requests.get(
            OPEN_METEO_FORECAST,
            params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
                "timezone": "Atlantic/Canary",
                "start_date": query.fecha, "end_date": query.fecha,
            },
            timeout=timeout,
        )
        fc.raise_for_status()
        daily = fc.json().get("daily", {})
        if not daily.get("time"):
            raise LookupError("La API no devolvió previsión para esa fecha.")
        code = int(daily["weather_code"][0])
        result = {
            "ok": True, "fuente": "Open-Meteo", "lugar": nombre, "fecha": query.fecha,
            "temp_min_c": daily["temperature_2m_min"][0],
            "temp_max_c": daily["temperature_2m_max"][0],
            "prob_precipitacion_pct": daily.get("precipitation_probability_max", [None])[0],
            "descripcion": WMO_CODES.get(code, f"código {code}"),
        }
        logger.info("get_weather OK | %s", result)
        return result
    except LookupError as exc:
        result = {"ok": False, "error": "lugar_no_encontrado", "mensaje": str(exc)}
        logger.warning("get_weather lugar | %s", result)
        return result
    except requests.Timeout:
        result = _simulated_forecast(query.fecha, query.lugar)
        result["mensaje"] = "La API tardó demasiado; estimación simulada."
        logger.warning("get_weather timeout -> simulado | %s", result)
        return result
    except requests.RequestException as exc:
        result = _simulated_forecast(query.fecha, query.lugar)
        result["mensaje"] = f"Fallo de red ({type(exc).__name__}); estimación simulada."
        logger.warning("get_weather red -> simulado | %s", result)
        return result


# --- Fábrica de la herramienta de LangChain --------------------------------- #
def build_weather_tool(use_real_api: bool = True):
    """Crea la herramienta LangChain ``get_weather`` enlazable al modelo.

    Args:
        use_real_api: Si ``False``, la herramienta usa el modo simulado.

    Returns:
        Una herramienta de LangChain (``@tool``) que el modelo puede invocar.
    """

    @tool("get_weather")
    def _get_weather(fecha: str, lugar: str) -> str:
        """Devuelve la previsión del tiempo para un lugar de Tenerife en una fecha.

        Úsala cuando el usuario pregunte por el tiempo o si va a llover. La fecha
        debe ir en formato ISO YYYY-MM-DD y el lugar es un sitio de Tenerife.
        La validación y el manejo de errores se hacen dentro (devuelve un texto
        útil aunque la fecha o el lugar no sean válidos).
        """
        result = get_weather(fecha, lugar, use_real_api=use_real_api)
        return json.dumps(result, ensure_ascii=False)

    return _get_weather
