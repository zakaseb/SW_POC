"""
One-time helper to cache all Hugging Face assets so the app can run offline.
Run this while you have internet connectivity.
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

from core.config import (
    MODEL_CACHE_DIR,
    RERANKER_LOCAL_PATH,
    RERANKER_MODEL_NAME,
    TOKENIZER_LOCAL_PATH,
    TOKENIZER_MODEL_NAME,
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


def main():
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using model cache: {MODEL_CACHE_DIR}")
    download_reranker()
    download_tokenizer()
    print("All offline assets cached. Remember to `ollama pull` the LLM/embedding tags you use.")


if __name__ == "__main__":
    main()
