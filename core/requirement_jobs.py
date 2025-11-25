import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Any

from rank_bm25 import BM25Okapi

from .config import REQUIREMENTS_OUTPUT_PATH
from .database import (
    create_requirement_job,
    update_requirement_job,
    get_requirement_job as db_get_requirement_job,
    get_latest_requirement_job as db_get_latest_requirement_job,
    list_requirement_jobs as db_list_requirement_jobs,
)
from .generation import generate_requirements_json, generate_excel_file, parse_requirements_payload
from .logger_config import get_logger

logger = get_logger(__name__)
OUTPUT_ROOT = Path(REQUIREMENTS_OUTPUT_PATH)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class RequirementJobManager:
    """Manages background requirement generation jobs."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="req-job")
        self._active_jobs: dict[str, Any] = {}
        logger.info("RequirementJobManager initialized with %s workers.", max_workers)

    @classmethod
    def instance(cls) -> "RequirementJobManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit_generation_job(
        self,
        *,
        user_id: int,
        language_model,
        requirements_chunks: List[Any],
        verification_docs: List[Any],
        general_docs: List[Any],
        top_k_general: int = 8,
        safe_char_cap: int = 32000,
    ) -> str:
        job_id = str(uuid.uuid4())
        metadata = {
            "top_k_general": top_k_general,
            "safe_char_cap": safe_char_cap,
            "requirements_chunk_count": len(requirements_chunks or []),
            "verification_doc_count": len(verification_docs or []),
            "general_doc_count": len(general_docs or []),
        }
        create_requirement_job(job_id, user_id, status="queued", metadata=metadata)
        logger.info("Queued requirement generation job %s for user %s", job_id, user_id)

        future = self._executor.submit(
            self._execute_job,
            job_id,
            user_id,
            language_model,
            requirements_chunks or [],
            verification_docs or [],
            general_docs or [],
            top_k_general,
            safe_char_cap,
        )
        self._active_jobs[job_id] = future
        future.add_done_callback(lambda _: self._active_jobs.pop(job_id, None))
        return job_id

    def _execute_job(
        self,
        job_id: str,
        user_id: int,
        language_model,
        requirements_chunks: List[Any],
        verification_docs: List[Any],
        general_docs: List[Any],
        top_k_general: int,
        safe_char_cap: int,
    ) -> None:
        logger.info("Starting requirement generation job %s", job_id)
        update_requirement_job(job_id, status="running")
        try:
            joiner = "\n\n---\n\n"
            verification_context_all = joiner.join(
                doc.page_content.strip()
                for doc in verification_docs
                if getattr(doc, "page_content", None)
            )

            general_paragraphs = _prepare_general_paragraphs(general_docs)
            tokenized_paragraphs = [_tokenize_text(p) for p in general_paragraphs] if general_paragraphs else []
            bm25 = BM25Okapi(tokenized_paragraphs) if tokenized_paragraphs else None

            verif_chars = len(verification_context_all)
            requirements_payload = []
            for chunk in requirements_chunks:
                chunk_text = getattr(chunk, "page_content", "") or ""
                general_context_selected = ""
                if bm25 and chunk_text.strip():
                    query_tokens = _tokenize_text(chunk_text)
                    if query_tokens:
                        scores = bm25.get_scores(query_tokens)
                        ranked_indices = sorted(
                            range(len(scores)),
                            key=lambda idx: scores[idx],
                            reverse=True,
                        )[:top_k_general]
                        top_general_chunks = [
                            general_paragraphs[idx]
                            for idx in ranked_indices
                            if general_paragraphs[idx].strip()
                        ]
                        general_context_selected = joiner.join(top_general_chunks)

                total_chars = verif_chars + len(general_context_selected) + len(chunk_text)
                if total_chars > safe_char_cap:
                    logger.warning(
                        "Job %s chunk exceeded safe char cap (%s > %s)",
                        job_id,
                        total_chars,
                        safe_char_cap,
                    )

                json_response = generate_requirements_json(
                    language_model,
                    chunk,
                    verification_methods_context=verification_context_all,
                    general_context=general_context_selected,
                )
                requirements_payload.append(json_response)

            excel_bytes = generate_excel_file(requirements_payload)
            parsed_requirements = parse_requirements_payload(requirements_payload)
            if not excel_bytes or not parsed_requirements:
                raise ValueError("Requirement generation produced no usable data.")

            job_dir = OUTPUT_ROOT / str(user_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            excel_path = job_dir / f"{job_id}.xlsx"
            excel_path.write_bytes(excel_bytes)

            json_path = job_dir / f"{job_id}.json"
            json_path.write_text(json.dumps(parsed_requirements, indent=2), encoding="utf-8")

            metadata = {
                "num_requirements": len(parsed_requirements),
                "json_path": str(json_path),
                "verif_chars": verif_chars,
                "general_paragraphs": len(general_paragraphs),
            }

            update_requirement_job(
                job_id,
                status="completed",
                result_path=str(excel_path),
                metadata=metadata,
            )
            logger.info("Requirement generation job %s completed.", job_id)
        except Exception as exc:
            logger.exception("Requirement generation job %s failed: %s", job_id, exc)
            update_requirement_job(job_id, status="failed", error_message=str(exc))


def _prepare_general_paragraphs(docs: List[Any], max_chars: int = 900, overlap: int = 120) -> List[str]:
    paragraphs: List[str] = []
    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        paragraphs.extend(_split_text(text, max_chars=max_chars, overlap=overlap))
    return paragraphs


def _split_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    cleaned = text.replace("\r", "\n")
    blocks = [block.strip() for block in cleaned.split("\n\n") if block.strip()]
    result: List[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            result.append(block)
        else:
            start = 0
            while start < len(block):
                end = min(start + max_chars, len(block))
                result.append(block[start:end].strip())
                start = max(end - overlap, end)
    return result


_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in _WORD_RE.findall(text)]


# Convenience exports
def submit_requirement_generation_job(**kwargs) -> str:
    return RequirementJobManager.instance().submit_generation_job(**kwargs)


def get_requirement_job(job_id: str):
    return db_get_requirement_job(job_id)


def get_latest_requirement_job(user_id: int):
    return db_get_latest_requirement_job(user_id)


def list_requirement_jobs(user_id: int, limit: int = 5):
    return db_list_requirement_jobs(user_id, limit)


def load_job_excel_bytes(job_info) -> bytes | None:
    if not job_info:
        return None
    path = job_info.get("result_path")
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_bytes()


def load_job_requirements(job_info):
    if not job_info:
        return None
    metadata = job_info.get("metadata") or {}
    json_path = metadata.get("json_path")
    if not json_path:
        return None
    file_path = Path(json_path)
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Failed to decode stored requirements JSON at %s", json_path)
        return None

