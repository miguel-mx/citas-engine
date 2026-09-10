"""Dependency probes for the dashboard's "Estado de servicios" panel.

The engine reports raw facts about the services *it* depends on — reachability,
latency, configured model — and leaves labels and presentation to Symfony. Every
probe is cheap, time-boxed and never raises: a failed probe is a result, not an
error, otherwise a single dead dependency would take the whole panel down.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import httpx

from app.config import settings

# Probes are user-facing (a page is waiting on them), so they fail fast rather
# than inheriting the 30 s budget the analysis pipeline uses.
PROBE_TIMEOUT = 4.0

# States Symfony knows how to render. `degraded` = reachable but not fully usable.
OK = "ok"
DEGRADED = "degraded"
DOWN = "down"
NOT_CONFIGURED = "not_configured"
# Configured but not verified: SIAB has a key, nobody has asked the service yet.
UNKNOWN = "unknown"


def _result(
    key: str,
    state: str,
    detail: str,
    latency_ms: int | None = None,
    message: str | None = None,
) -> dict:
    return {
        "key": key,
        "state": state,
        "detail": detail,
        "latency_ms": latency_ms,
        "message": message,
    }


async def _probe_openalex(client: httpx.AsyncClient) -> dict:
    """One cheapest-possible work fetch: proves the API answers and we're in the
    polite pool. 429 means we're being throttled — usable, but degraded."""
    host = httpx.URL(settings.openalex_base).host or settings.openalex_base
    started = time.perf_counter()

    try:
        response = await client.get(
            f"{settings.openalex_base}/works",
            params={"per_page": 1, "select": "id", "mailto": settings.mailto},
        )
    except httpx.HTTPError as e:
        return _result("openalex", DOWN, host, message=f"sin conexión: {type(e).__name__}")

    latency = int((time.perf_counter() - started) * 1000)

    if response.status_code == 429:
        return _result("openalex", DEGRADED, host, latency, "límite de peticiones alcanzado")
    if response.status_code >= 400:
        return _result("openalex", DOWN, host, latency, f"HTTP {response.status_code}")

    return _result("openalex", OK, host, latency)


async def _probe_ollama(client: httpx.AsyncClient, base: str, model: str) -> dict:
    """Reachability *and* whether the configured model is actually pulled — a
    running Ollama without the model still cannot write a report.

    `base`/`model` are whatever the caller is actually going to use (SIAB passes its
    own admin-configured values), so the dashboard cannot report a healthy server
    that reports will never be sent to.
    """
    # Ollama often runs on another machine (e.g. a shared GPU box at the institute),
    # so the detail names the host unless it really is this machine.
    url = httpx.URL(base)
    port = f":{url.port}" if url.port else ""
    host = "" if url.host in {"localhost", "127.0.0.1", "::1"} else url.host
    detail = f"{model} · {host}{port}" if (host or port) else f"{model} · {base}"
    started = time.perf_counter()

    try:
        response = await client.get(f"{base}/api/tags")
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as e:
        return _result("ollama", DOWN, detail, message=f"sin conexión: {type(e).__name__}")
    except ValueError:
        return _result("ollama", DOWN, detail, message="respuesta ilegible")

    latency = int((time.perf_counter() - started) * 1000)
    installed = [m.get("name", "") for m in payload.get("models", [])]

    if model not in installed:
        return _result(
            "ollama", DEGRADED, detail, latency,
            f"el modelo {model} no está descargado (ollama pull {model})",
        )

    return _result("ollama", OK, detail, latency)


# ── Verifying one API key ─────────────────────────────────────────────────────
# The cheapest authenticated request each service offers. The point is not the
# result but the status code: whether the key is accepted at all.
_KEY_PROBES: dict[str, tuple[str, dict, str]] = {
    "scopus": (
        "https://api.elsevier.com/content/search/scopus",
        {"query": "AU-ID(6602738988)", "count": 1},
        "X-ELS-APIKey",
    ),
    "wos": (
        "https://api.clarivate.com/apis/wos-starter/v1/documents",
        {"q": "TS=(topology)", "limit": 1},
        "X-ApiKey",
    ),
}


async def probe_source_key(source: str, api_key: str) -> dict:
    """Ask the service whether it accepts this key, and say plainly what it answered.

    Exists because the only way to learn a key was rejected used to be to launch an
    analysis and read the warnings twenty minutes later. The key is never logged or
    echoed back — only the verdict is.
    """
    probe = _KEY_PROBES.get(source)

    if probe is None:
        return {"ok": False, "state": DOWN, "message": f"Fuente desconocida: {source}."}

    if not api_key.strip():
        return {"ok": False, "state": NOT_CONFIGURED, "message": "No hay clave guardada para esta fuente."}

    url, params, header = probe

    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url, params=params, headers={header: api_key, "Accept": "application/json"})
    except httpx.HTTPError as e:
        return {"ok": False, "state": DOWN, "message": f"No se pudo contactar con el servicio: {type(e).__name__}."}

    code = response.status_code

    if code == 200:
        return {"ok": True, "state": OK, "message": "La clave funciona: el servicio respondió correctamente."}
    if code in (401, 403):
        return {
            "ok": False,
            "state": DOWN,
            "message": (
                f"El servicio rechazó la clave ({code}). Suele significar una clave de otro producto "
                "—esta consulta usa wos-starter— o una suscripción caducada o no activada."
                if source == "wos"
                else f"El servicio rechazó la clave ({code}). Revisa que sea una clave de Scopus Search vigente."
            ),
        }
    if code == 429:
        return {"ok": False, "state": DEGRADED, "message": "La clave es válida pero la cuota está agotada (429)."}

    return {"ok": False, "state": DOWN, "message": f"Respuesta inesperada del servicio ({code})."}


def _optional_source(key: str, has_key: bool) -> dict:
    """Scopus / WoS in the panel.

    Whether a key exists is the caller's business, not this service's: SIAB stores
    them (encrypted) and sends them per analysis, so the engine's own .env says
    nothing about it. Reading that .env here is what made the panel report "sin
    clave" for a Scopus key that works.

    Configuration only, deliberately. Verifying a key means an authenticated request
    to a metered service; doing that on every dashboard refresh would burn quota to
    tell an administrator something the "Probar clave" button answers on demand.
    """
    if not has_key:
        return _result(key, NOT_CONFIGURED, "fuente opcional", message="sin clave")

    return _result(
        key,
        UNKNOWN,
        "fuente opcional",
        message="clave configurada · pruébala en Configuración del motor",
    )


async def probe_all(
    ollama_base: str | None = None,
    ollama_model: str | None = None,
    scopus_key: bool | None = None,
    wos_key: bool | None = None,
) -> dict:
    """Run every probe concurrently. Total wall time ≈ the slowest probe.

    scopus_key / wos_key say whether the *caller* holds a key for those sources.
    Booleans rather than the keys themselves: this is a GET, and a secret in a query
    string ends up in access logs.
    """
    base = ollama_base or settings.ollama_base
    model = ollama_model or settings.model

    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT, follow_redirects=True) as client:
        openalex, ollama = await asyncio.gather(
            _probe_openalex(client),
            _probe_ollama(client, base, model),
        )

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "services": [
            openalex,
            ollama,
            _optional_source("scopus", settings.use_scopus if scopus_key is None else scopus_key),
            _optional_source("wos", settings.use_wos if wos_key is None else wos_key),
        ],
    }
