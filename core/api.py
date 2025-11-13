from typing import Optional
import time
import logging

from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
import httpx

from typing import Optional, Any, Dict, Union
from pydantic import BaseModel, Field

from core.config import OLLAMA_URL
from core.logger_config import setup_logging, get_logger  # your existing logger config


# 🔧 initialize logging for this process
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="LLM Wrapper", version="0.1")


# =========================
# 💾 Pydantic models
# =========================

class GenerateReq(BaseModel):
    prompt: str
    model: str = "mistral:7b"
    temperature: float = 0.2
    max_tokens: int | None = None


class GenerateResp(BaseModel):
    model: str
    response: str


class EmbedReq(BaseModel):
    texts: list[str]
    model: str = "mistral:7b"


class EmbedResp(BaseModel):
    model: str
    embeddings: list[list[float]]


# =========================
# 🌐 Middleware: log ALL HTTP requests
# =========================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    method = request.method
    path = request.url.path
    query = str(request.url.query)

    logger.info(f"➡️ {method} {path}?{query}")

    body_bytes = b""
    if method in ("POST", "PUT", "PATCH"):
        body_bytes = await request.body()
        preview = body_bytes.decode("utf-8", errors="ignore")[:500]
        logger.info(f"   Request body preview: {preview!r}")

        # Re-inject body so FastAPI can still read it
        async def receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"⬅️ {method} {path} completed in {process_time:.2f}ms "
        f"with status {response.status_code}"
    )

    return response


# =========================
# 🚀 Startup
# =========================

@app.on_event("startup")
async def on_startup():
    logger.info(f"🚀 LLM Wrapper starting. Using Ollama at {OLLAMA_URL}")


# =========================
# 🔎 Basic endpoints
# =========================

@app.get("/")
async def root():
    logger.info("GET / called")
    return {
        "ok": True,
        "service": "LLM Wrapper",
        "endpoints": ["/health", "/generate", "/embed"],
    }


@app.get("/health")
async def health():
    logger.info("Health check requested")
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
        ok = r.status_code == 200
        logger.info(f"Health -> {ok}")
        return {"ok": ok}
    except Exception as e:
        logger.exception("Health check failed")
        return {"ok": False, "error": str(e)}


# =========================
# ✨ /generate
# =========================

@app.post("/generate", response_model=GenerateResp)
async def generate(body: GenerateReq):
    logger.info(
        f"/generate called | model={body.model} | temp={body.temperature} | "
        f"max_tokens={body.max_tokens} | prompt_preview={body.prompt[:80]!r}"
    )

    options = {"temperature": body.temperature, "num_ctx": 8192}
    if body.max_tokens is not None:
        options["num_predict"] = body.max_tokens

    payload = {
        "model": body.model,
        "prompt": body.prompt,
        "stream": False,
        "options": options,
    }

    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"{OLLAMA_URL}/api/generate", json=payload)

    if r.status_code != 200:
        # surface Ollama’s message so you can see "context length exceeded" etc.
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        logger.error(f"/generate upstream error: {detail}")
        raise HTTPException(status_code=502, detail={"upstream": detail})

    data = r.json()
    logger.info("/generate succeeded")
    return GenerateResp(model=body.model, response=data.get("response", ""))


@app.get("/generate")
async def generate_get(
    prompt: Optional[str] = Query(default=None),
    model: str = "mistral:7b",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
):
    """
    - GET /generate                 -> info stub (for GUI/health checks)
    - GET /generate?prompt=hello    -> real generate call
    """
    if prompt is None:
        logger.info("GET /generate called with no prompt - returning info stub")
        return {
            "ok": True,
            "message": "LLM /generate endpoint is alive. "
                       "Use POST /generate or GET /generate?prompt=...",
            "expected_query_params": ["prompt", "model", "temperature", "max_tokens"],
        }

    logger.info(
        f"GET /generate called | model={model} | temp={temperature} | "
        f"max_tokens={max_tokens} | prompt_preview={prompt[:80]!r}"
    )

    body = GenerateReq(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await generate(body)


# =========================
# 📏 /embed
# =========================

@app.post("/embed", response_model=EmbedResp)
async def embed(body: EmbedReq):
    logger.info(f"/embed called | model={body.model} | n_texts={len(body.texts)}")
    out: list[list[float]] = []

    async with httpx.AsyncClient(timeout=60.0) as c:
        for t in body.texts:
            r = await c.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": body.model, "prompt": t},
            )
            if r.status_code != 200:
                logger.error(f"/embed upstream error: {r.text}")
                raise HTTPException(r.status_code, r.text)
            out.append(r.json()["embedding"])

    logger.info("/embed succeeded")
    return EmbedResp(model=body.model, embeddings=out)


@app.get("/embed")
async def embed_get(
    texts: Optional[list[str]] = Query(default=None),
    model: str = "mistral:7b",
):
    """
    - GET /embed                            -> info stub
    - GET /embed?texts=hello&texts=world    -> real embed call
    """
    if texts is None:
        logger.info("GET /embed called with no texts - returning info stub")
        return {
            "ok": True,
            "message": "LLM /embed endpoint is alive. "
                       "Use POST /embed or GET /embed?texts=...",
            "expected_query_params": ["texts", "model"],
        }

    logger.info(f"GET /embed called | model={model} | n_texts={len(texts)}")

    body = EmbedReq(texts=texts, model=model)
    return await embed(body)

class UIEvent(BaseModel):
    event: str
    user_id: Optional[Union[int, str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.post("/ui-event")
async def ui_event(ev: UIEvent):
    """
    Receive UI events from the Streamlit app so they show up in API logs.
    """
    logger.info(
        f"📲 UI_EVENT | event={ev.event} | user={ev.user_id} | metadata={ev.metadata}"
    )
    return {"ok": True}
