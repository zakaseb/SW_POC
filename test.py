#!/usr/bin/env python3
"""
Load → chunk → index 'Verification Methods.docx' using ONLY LM Studio embeddings.
- Ensures /v1/embeddings gets list[str]
- Stores any list/dict metadata as JSON strings (so nothing is lost)
- Disables ctx-length check for local servers
- In-memory Chroma (no disk persistence)
- Prints chunk & metadata previews
"""

import os
import sys
import json
from pathlib import Path
import shutil
from typing import Any, List, Tuple

for k in ["OPENAI_PROXY","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY",
          "http_proxy","https_proxy","all_proxy"]:
    os.environ.pop(k, None)

# ----------------------------
# LM Studio endpoint + models
# ----------------------------
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1234/v1")    # LM Studio local server
os.environ.setdefault("OPENAI_API_KEY", "lm-studio")                     # dummy key is fine

# ⚠️ Use the *exact* id shown by LM Studio GET /v1/models (often "bge-m3")
os.environ.setdefault("LM_STUDIO_EMBEDDING_MODEL_NAME", "bge-m3")
os.environ.setdefault("LM_STUDIO_LLM_NAME", "mistral-7b-instruct-v0.3")

# ----------------------------
# Project path so core.* imports resolve
# ----------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/home/nvidia/LLM_integration/SW_POC")).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------
# Streamlit shim (minimal) because your core.* uses st.*
# ----------------------------
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    import types
    st = types.SimpleNamespace()
    st.session_state = {}
    def _noop(*args, **kwargs): pass
    st.warning = _noop
    st.error = _noop
    st.write = _noop
    st.info = _noop
    st.success = _noop
    sys.modules["streamlit"] = st

# ----------------------------
# Your project imports
# ----------------------------
from core.config import CONTEXT_PDF_STORAGE_PATH
from core.document_processing import load_document, chunk_documents
from core.logger_config import get_logger
logger = get_logger(__name__)

# ----------------------------
# Paths (matches your app’s pattern)
# ----------------------------
CONTEXT_DIR = Path(CONTEXT_PDF_STORAGE_PATH).resolve()
APP_ROOT = PROJECT_ROOT
TEMPLATE_DOC = (APP_ROOT / "Verification Methods.docx").resolve()

CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
dest = CONTEXT_DIR / TEMPLATE_DOC.name
if not TEMPLATE_DOC.exists():
    raise FileNotFoundError(f"Put 'Verification Methods.docx' at: {TEMPLATE_DOC}")
if not dest.exists():
    shutil.copyfile(TEMPLATE_DOC, dest)
    logger.info(f"Global default context copied to: {dest}")
else:
    logger.info(f"Global default context already present: {dest}")

# ----------------------------
# Embeddings (LM Studio / OpenAI-compatible)
# ----------------------------
try:
    from langchain_openai import OpenAIEmbeddings   # preferred
except Exception:
    from langchain_community.embeddings import OpenAIEmbeddings  # fallback

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

EMB_NAME = os.getenv("LM_STUDIO_EMBEDDING_MODEL_NAME", "bge-m3")   # must exist in LM Studio
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")

embeddings = OpenAIEmbeddings(
    model=EMB_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    check_embedding_ctx_length=False,   # helpful with some local servers
)

# Two in-memory Chroma collections (no persist_directory)
st.session_state.CONTEXT_VECTOR_DB = Chroma(embedding_function=embeddings)
st.session_state.PERSISTENT_VECTOR_DB = Chroma(embedding_function=embeddings)

# ----------------------------
# Helpers: coerce + JSON-sanitize
# ----------------------------
def to_texts_and_metas(chunks: List[Any]) -> Tuple[List[str], List[dict]]:
    """Turn mixed chunk objects into (list[str], list[dict]) for indexing."""
    texts, metas = [], []
    for ch in chunks:
        # Already a Document
        if isinstance(ch, Document):
            t = ch.page_content
            m = ch.metadata or {}
        # String
        elif isinstance(ch, str):
            t = ch
            m = {}
        # Dict-like from chunkers (docling, etc.)
        elif isinstance(ch, dict):
            t = ch.get("page_content") or ch.get("text") or ch.get("content")
            m = ch.get("metadata") or {k: v for k, v in ch.items() if k not in {"page_content", "text", "content"}}
        else:
            # Fallback: stringify
            t, m = str(ch), {}

        if t is None:
            t = ""
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if not t:
            continue

        texts.append(t)
        metas.append(m)
    return texts, metas

ALLOWED = (str, int, float, bool, type(None))

def sanitize_metadatas_json(metadatas: List[dict], n_texts: int) -> List[dict]:
    """
    Convert metadata values to types accepted by vector stores, while preserving structure:
      - lists/tuples/sets → JSON string
      - dicts            → JSON string
      - other unsupported → str(v)
    """
    if not metadatas:
        return [{} for _ in range(n_texts)]

    out_all: List[dict] = []
    for md in metadatas:
        md = md or {}
        out: dict = {}
        for k, v in md.items():
            if isinstance(v, (list, tuple, set)):
                try:
                    v = json.dumps(list(v), ensure_ascii=False)
                except Exception:
                    v = str(list(v))
            elif isinstance(v, dict):
                try:
                    v = json.dumps(v, ensure_ascii=False)
                except Exception:
                    v = str(v)
            elif not isinstance(v, ALLOWED):
                v = str(v)
            out[k] = v
        out_all.append(out)

    # pad/truncate to match texts
    if len(out_all) < n_texts:
        out_all.extend({} for _ in range(n_texts - len(out_all)))
    elif len(out_all) > n_texts:
        out_all = out_all[:n_texts]
    return out_all

def index_texts(texts: List[str], metadatas: List[dict], vector_db: Chroma) -> None:
    """Strict indexing path that guarantees LM Studio receives list[str] and metadata are scalars."""
    # Embedding input guardrails
    assert all(isinstance(t, str) for t in texts), "Non-string found in texts"
    assert all(t.strip() for t in texts), "Empty text chunk present"

    # JSON-preserving metadata sanitizer
    metadatas = sanitize_metadatas_json(metadatas, n_texts=len(texts))

    # Final guard — vector stores require scalar values
    for i, md in enumerate(metadatas):
        for k, v in md.items():
            if not isinstance(v, (str, int, float, bool, type(None))):
                raise ValueError(f"Bad metadata at idx {i} key '{k}': {type(v).__name__} -> {v!r}")

    # Index using add_texts so only strings hit embeddings
    vector_db.add_texts(texts=texts, metadatas=metadatas)

# ----------------------------
# RUN: load → chunk → coerce → PREVIEW → index
# ----------------------------
ctx_file = dest
mtime = ctx_file.stat().st_mtime

raw = load_document(str(ctx_file))
if not raw:
    print("No content loaded; nothing to index.")
    sys.exit(0)

# Your own chunker (uses your project's logic)
_, _, chunks = chunk_documents(raw, str(CONTEXT_DIR), classify=False)

if chunks:
    texts, metas = to_texts_and_metas(chunks)

    # ---- Preview (raw) ----
    PREVIEW_N = 5
    print(f"Prepared {len(texts)} chunks. Showing first {min(PREVIEW_N, len(texts))}:")
    for i in range(min(PREVIEW_N, len(texts))):
        t = texts[i]
        m = metas[i] if i < len(metas) else {}
        print(f"\n--- Chunk #{i} ---")
        print(f"len(text) = {len(t)}")

        # FIX: compute pieces first (avoid backslash inside f-string expression)
        preview = t[:200].replace("\n", " ")
        suffix = "..." if len(t) > 200 else ""
        print(f"text     = {preview}{suffix}")

        print("metadata =", json.dumps(m, ensure_ascii=False, indent=2))

    # ---- Preview (sanitized) ----
    metas_sanitized = sanitize_metadatas_json(metas, n_texts=len(texts))
    print("\nSanitized metadata preview:")
    print(json.dumps(metas_sanitized[:PREVIEW_N], ensure_ascii=False, indent=2))

    # ---- Index both stores ----
    index_texts(texts, metas, st.session_state.CONTEXT_VECTOR_DB)
    index_texts(texts, metas, st.session_state.PERSISTENT_VECTOR_DB)  # backup

    logger.info(f"Global context indexed: {len(texts)} chunks.")
    st.session_state.global_ctx_indexed_mtime = mtime
    st.session_state.context_document_loaded = True
    print("✅ Indexed: True")
else:
    print("No chunks produced; nothing to index.")

# ---- Find chunks around a target string ----
def show_chunks_around(texts, metas, needle: str, window: int = 1, preview_chars: int = 220):
    needle_lc = needle.lower()
    hits = [i for i, t in enumerate(texts) if needle_lc in t.lower()]
    if not hits:
        print(f"No chunks contain: {needle!r}")
        return

    print(f"\nFound {len(hits)} hit(s) for {needle!r}. Showing ±{window} neighbor(s):")
    for idx in hits:
        start = max(0, idx - window)
        end   = min(len(texts) - 1, idx + window)

        print("\n" + "="*72)
        print(f"Hit at chunk #{idx} (neighbors {start}..{end})")
        for j in range(start, end + 1):
            t = texts[j]
            m = metas[j] if j < len(metas) else {}
            # compact one-line preview per chunk
            preview = t[:preview_chars].replace("\n", " ")
            suffix = "..." if len(t) > preview_chars else ""
            flag = " (HIT)" if j == idx else ""
            print(f"\n--- Chunk #{j}{flag} ---")
            print(f"len(text) = {len(t)}")
            print(f"text      = {preview}{suffix}")
            if m:
                try:
                    print("metadata  =", json.dumps(m, ensure_ascii=False))
                except Exception:
                    print("metadata  =", str(m))

# Call it (e.g., looking for 'bimg')
show_chunks_around(texts, metas, needle="bimg", window=1, preview_chars=220)
