"""
One-time helper to cache all Hugging Face assets so the app can run offline.
Run this while you have internet connectivity.
"""

from pathlib import Path

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


def main():
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using model cache: {MODEL_CACHE_DIR}")
    download_reranker()
    download_tokenizer()
    download_docling_layout_models()
    print("All offline assets cached. Remember to `ollama pull` the LLM/embedding tags you use.")


if __name__ == "__main__":
    main()
