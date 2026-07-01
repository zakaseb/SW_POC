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
from .heading_cleanup import headings_before_first_numbered, is_numbered_heading
from .source_matching import best_source_sentence
from .logger_config import get_logger
from core.jama_hierarchy import build_hierarchy_workbook_bytes

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

    def active_job_ids(self) -> set:
        """Job ids with a live worker future in THIS process."""
        return set(self._active_jobs.keys())

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

            # Determine front-matter headings (those before the first numbered
            # heading) from the union of all chunk heading paths, so we can skip
            # chunks that belong to front matter (cover, foreword, ToC, etc.).
            all_headings_ordered = []
            seen_h = set()
            for chunk in requirements_chunks:
                for h in (getattr(chunk, "metadata", {}) or {}).get("headings") or []:
                    if h not in seen_h:
                        seen_h.add(h)
                        all_headings_ordered.append(h)
            pre_numbered = set(headings_before_first_numbered(all_headings_ordered))
            # Only treat headings as front matter when the document actually has
            # numbered sections. With native Docling conversion every chunk shares
            # a non-numbered document-title root, and unnumbered documents would
            # otherwise have *every* heading classified as front matter.
            has_numbered_sections = any(
                is_numbered_heading(h) for h in all_headings_ordered
            )

            verif_chars = len(verification_context_all)
            requirements_payload = []
            chunk_results = []   # list of (headings_path, parsed_requirements)
            skipped_front_matter = 0

            for chunk in requirements_chunks:
                headings_path = (getattr(chunk, "metadata", {}) or {}).get("headings") or []

                # Skip front-matter chunks. A chunk is front matter when its
                # deepest (leaf) heading is one of the pre-numbered front-matter
                # sections (cover, approvals, ToC, etc.). Checking the leaf rather
                # than the root avoids misclassifying real requirement chunks that
                # share a non-numbered document-title root.
                leaf_heading = headings_path[-1] if headings_path else ""
                if has_numbered_sections and leaf_heading in pre_numbered:
                    skipped_front_matter += 1
                    continue

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
                # Pair each parsed requirement group with its heading path so the
                # hierarchy workbook can indent it under the right section.
                parsed_chunk_reqs = parse_requirements_payload([json_response])
                # Trace every requirement back to the exact chunk and sentence it
                # was extracted from. Duplication across rows is expected: several
                # requirements may originate from the same chunk/sentence.
                for req in parsed_chunk_reqs:
                    if not isinstance(req, dict):
                        continue
                    source_sentence, conf_score = best_source_sentence(
                        req.get("Description", ""), chunk_text
                    )
                    req["source_chunk"] = chunk_text
                    req["source_sentence"] = source_sentence
                    req["conf_score"] = conf_score
                chunk_results.append((headings_path, parsed_chunk_reqs))

            logger.info(
                "Job %s: processed %d chunk(s), skipped %d front-matter chunk(s).",
                job_id,
                len(chunk_results),
                skipped_front_matter,
            )

            # Build the indented Jama hierarchy workbook from (headings, reqs) pairs.
            excel_bytes = build_hierarchy_workbook_bytes(chunk_results)
            parsed_requirements = [
                req for _, reqs in chunk_results for req in (reqs or [])
            ]

            # excel_bytes = generate_excel_file(requirements_payload)
            # parsed_requirements = parse_requirements_payload(requirements_payload)
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
            safe_overlap = max(0, overlap)
            step = max(1, max_chars - safe_overlap)  # always progress, even if overlap >= max_chars
            while start < len(block):
                end = min(start + max_chars, len(block))
                result.append(block[start:end].strip())
                if end == len(block):
                    break
                start += step
    return result


_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in _WORD_RE.findall(text)]


# Convenience exports
def submit_requirement_generation_job(**kwargs) -> str:
    return RequirementJobManager.instance().submit_generation_job(**kwargs)


def is_job_active(job_id: str) -> bool:
    """True if the job has a live worker future in the current process."""
    if not job_id:
        return False
    return job_id in RequirementJobManager.instance().active_job_ids()


def reconcile_stale_jobs(user_id: int) -> int:
    """Mark jobs left in 'queued'/'running' by a previous (now-dead) process as failed.

    Requirement jobs run inside the Streamlit process via an in-memory thread pool.
    If the process is restarted while a job is in flight, the DB row stays 'running'
    forever even though no worker exists. The UI then polls that status on a 3s loop
    indefinitely. Here we detect such rows (queued/running but with no live worker in
    THIS process) and fail them so the UI stops polling. Returns the count reconciled.
    """
    if not user_id:
        return 0
    active = RequirementJobManager.instance().active_job_ids()
    reconciled = 0
    for job in db_list_requirement_jobs(user_id, limit=50):
        if job.get("status") in {"queued", "running"} and job.get("id") not in active:
            update_requirement_job(
                job["id"],
                status="failed",
                error_message="Job was interrupted (application restarted before it finished). Please run it again.",
            )
            reconciled += 1
    if reconciled:
        logger.info("Reconciled %d stale requirement job(s) for user %s.", reconciled, user_id)
    return reconciled


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

