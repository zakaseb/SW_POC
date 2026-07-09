"""
One-time helper to cache all Hugging Face assets so the app can run offline.
Run this while you have internet connectivity.
"""

from pathlib import Path
import requests
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from docling.utils.model_downloader import download_models as download_docling_models

from core.config import (
    MODEL_CACHE_DIR,
    RERANKER_LOCAL_PATH,
    RERANKER_MODEL_NAME,
    TOKENIZER_LOCAL_PATH,
    TOKENIZER_MODEL_NAME,
    DOCLING_ARTIFACTS_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_NAME,
    OLLAMA_EMBEDDING_MODEL_NAME,
)


def _safe_snapshot(model_name: str, target_dir: Path) -> None:
    """
    Download a HF repository to a deterministic local folder.
    """
    print(f"→ Downloading {model_name} to {target_dir} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_dir),
        local_dir_use_symlinks=True,
        ignore_patterns=["*.msgpack", "*.h5"],  # keep cache smaller
    )


def download_reranker():
    _safe_snapshot(RERANKER_MODEL_NAME, RERANKER_LOCAL_PATH)
    # Force load so sentence-transformers writes any extra metadata
    CrossEncoder(
        str(RERANKER_LOCAL_PATH),
        cache_folder=str(MODEL_CACHE_DIR),
        local_files_only=False,
    )
    print("✓ Reranker cached.")


def download_tokenizer():
    _safe_snapshot(TOKENIZER_MODEL_NAME, TOKENIZER_LOCAL_PATH)
    AutoTokenizer.from_pretrained(
        str(TOKENIZER_LOCAL_PATH),
        cache_dir=str(MODEL_CACHE_DIR),
    )
    print("✓ Docling tokenizer cached.")


def download_docling_layout_models():
    """
    Cache the Docling PDF pipeline models (layout detection + OCR) so
    DocumentConverter().convert() can run on PDFs with HF_HUB_OFFLINE=1.
    Table structure / picture classifier / code-formula models are skipped
    since chunk_documents() only needs layout (for headings) and OCR.
    """
    print(f"→ Downloading Docling layout/OCR models to {DOCLING_ARTIFACTS_PATH} ...")
    DOCLING_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    download_docling_models(
        output_dir=DOCLING_ARTIFACTS_PATH,
        force=True,      # don't silently skip if the dir already exists but is incomplete
        progress=True,   # show download bars instead of a silent hang/no-op
        with_layout=True,
        with_tableformer=True,
        with_easyocr=True,
        with_code_formula=False,
        with_picture_classifier=False,
    )

    # Verify the files we actually depend on are present before declaring success.
    easyocr_dir = DOCLING_ARTIFACTS_PATH / "EasyOcr"
    required_easyocr_files = ["craft_mlt_25k.pth", "english_g2.pth", "latin_g2.pth"]
    missing = [f for f in required_easyocr_files if not (easyocr_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Docling layout/OCR download reported success but is missing "
            f"EasyOcr files: {missing} under {easyocr_dir}. Re-run with force=True "
            f"or check network connectivity."
        )

    print("✓ Docling layout/OCR models cached.")

def _ollama_installed_tags(base_url: str) -> set[str]:
    """
    Returns the set of model tags currently pulled in the local Ollama instance.
    Raises if Ollama itself isn't reachable (distinct from "model missing").
    """
    resp = requests.get(f"{base_url}/api/tags", timeout=5)
    resp.raise_for_status()
    return {m["name"] for m in resp.json().get("models", [])}


def _ollama_pull(base_url: str, model_name: str) -> None:
    """
    Streams `ollama pull <model_name>` via the HTTP API and prints progress.
    """
    print(f"→ Pulling Ollama model '{model_name}' ...")
    with requests.post(
        f"{base_url}/api/pull",
        json={"model": model_name, "stream": True},
        stream=True,
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        last_status = None
        for line in resp.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            status = event.get("status")
            if status and status != last_status:
                print(f"    {status}")
                last_status = status
            if event.get("error"):
                raise RuntimeError(f"Ollama pull failed for '{model_name}': {event['error']}")
    print(f"✓ Ollama model '{model_name}' ready.")


def ensure_ollama_models():
    """
    Ensures every Ollama model tag the app depends on (LLM + embedding) is
    pulled locally. Pulls whatever is missing; no-ops for tags already present.
    Skips gracefully (with a warning) if Ollama isn't reachable, since this
    script may run before Ollama is started.
    """
    required = {OLLAMA_LLM_NAME, OLLAMA_EMBEDDING_MODEL_NAME}

    try:
        installed = _ollama_installed_tags(OLLAMA_BASE_URL)
    except Exception as e:
        print(
            f"⚠ Could not reach Ollama at {OLLAMA_BASE_URL} to check installed models "
            f"({e}). Skipping model pull — make sure to run "
            f"`ollama pull {'` and `ollama pull '.join(sorted(required))}` manually."
        )
        return

    for model_name in sorted(required):
        # Ollama tags are often stored with an implicit ":latest" suffix.
        candidates = {model_name, f"{model_name}:latest"} if ":" not in model_name else {model_name}
        if installed & candidates:
            print(f"✓ Ollama model '{model_name}' already present.")
            continue
        _ollama_pull(OLLAMA_BASE_URL, model_name)

def main():
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using model cache: {MODEL_CACHE_DIR}")
    download_reranker()
    download_tokenizer()
    download_docling_layout_models()
    ensure_ollama_models()
    print("All offline assets cached.")


if __name__ == "__main__":
    main()
