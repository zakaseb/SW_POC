#!/usr/bin/env python3
"""
Performance Metrics Comparison: AI-Generated vs Human-Made Requirement Excel Outputs

Compares the BULK of AI-generated requirements against the BULK of human-created
reference (gold standard). No row-wise matching by ID—metrics compare the two
datasets as wholes.

Usage:
    1. Set GENERATED_EXCEL_PATH and HUMAN_EXCEL_PATH below (or pass as arguments)
    2. Run: python compare_requirements_performance.py
           or: python compare_requirements_performance.py <generated.xlsx> <human.xlsx>

Output:
    1) compare_requirements_metrics_<timestamp>.xlsx
       - Summary: Bulk performance metrics
       - Distributions: VerificationMethod, RequirementType, Tags comparison
       - Metrics_Explanation: Description of each metric and what it measures

    2) <generated-stem>_with_verbatim_<timestamp>.xlsx (only when --source /
       SOURCE_DOCUMENT_PATH is provided):
       A copy of the generated requirements table with two extra columns
       appended to the main output table (no separate sheet):
         - "Source Text (Verbatim)": the contiguous span from the original
           input document that best matches the requirement's Description.
         - "Source Match Score": similarity in [0, 1] between that verbatim
           span and the Description (1.0 = exact match). Some rows may have
           no acceptable match and will leave the verbatim cell empty.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from datetime import datetime
from zipfile import BadZipFile

import pandas as pd

# =============================================================================
# CONFIGURATION: Set your file paths here
# =============================================================================
# Path to the AI-generated requirements Excel output
GENERATED_EXCEL_PATH = "path/to/generated_requirements.xlsx"

# Path to the human-made (reference/gold standard) requirements Excel
HUMAN_EXCEL_PATH = "path/to/human_requirements.xlsx"

# Path for the output metrics Excel (optional; default: timestamped in current dir)
OUTPUT_EXCEL_PATH = None

# Path to the ORIGINAL input document the AI extracted requirements from
# (.pdf, .docx, .txt). When provided, a second Excel file is written next to
# OUTPUT_EXCEL_PATH: a copy of the generated requirements table with two
# extra columns ("Source Text (Verbatim)", "Source Match Score") appended.
# Leave as "path/to/source_document.pdf" (or None) to skip verbatim mapping.
SOURCE_DOCUMENT_PATH = "path/to/source_document.pdf"

# Optional override for the enriched-output Excel path. When None and a
# source document is provided, the file is derived from the generated path
# (e.g. generated_requirements_with_verbatim_<timestamp>.xlsx).
ENRICHED_OUTPUT_PATH = None

# =============================================================================


def _normalize_for_compare(val):
    """Normalize field value for comparison."""
    if pd.isna(val):
        return ""
    if isinstance(val, list):
        return ", ".join(str(x).strip() for x in val if x)
    return str(val).strip()


def _text_similarity(a: str, b: str) -> float:
    """Return similarity ratio in [0, 1] using SequenceMatcher."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _word_set(text: str) -> set:
    """Extract unique words (lowercased) from text."""
    if not text or not str(text).strip():
        return set()
    return set(re.findall(r"[a-zA-Z0-9]+", str(text).lower()))


def _tags_to_set(tags_val) -> set:
    """Parse tags field into set of lowercase tag strings."""
    if pd.isna(tags_val) or not str(tags_val).strip():
        return set()
    parts = re.split(r"[,;\s]+", str(tags_val).strip())
    return {p.strip().lower() for p in parts if p.strip()}


def load_requirements(path: str) -> pd.DataFrame:
    """Load requirements from an Excel or CSV file. Expects first sheet to contain requirements."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        elif suffix == ".xls":
            try:
                df = pd.read_excel(path, engine="xlrd", sheet_name=0)
            except ImportError:
                raise ValueError(
                    "Reading .xls requires xlrd. Install with: pip install xlrd. "
                    "Or save the file as .xlsx in Excel."
                )
        else:
            df = pd.read_excel(path, engine="openpyxl", sheet_name=0)
    except BadZipFile as e:
        raise ValueError(
            f"'{path}' is not a valid .xlsx file. Common causes:\n"
            "  - File is .xls (old Excel): save as .xlsx or install xlrd\n"
            "  - File is CSV: rename to .csv or save as .xlsx from Excel\n"
            "  - File is corrupted or empty"
        ) from e
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {
        "page number": "Page Number",
        "page_number": "Page Number",
        "section": "Section",
        "verificationmethod": "VerificationMethod",
        "requirementtype": "RequirementType",
        "documentrequirementid": "DocumentRequirementID",
    }
    for old, new in col_map.items():
        for c in list(df.columns):
            if c.lower() == old:
                df = df.rename(columns={c: new})
                break
    return df


def _field_completeness(df: pd.DataFrame, field: str) -> float:
    """Fraction of rows with non-empty value for field."""
    if field not in df.columns:
        return 0.0
    n = len(df)
    if n == 0:
        return 0.0
    non_empty = sum(1 for v in df[field] if _normalize_for_compare(v))
    return non_empty / n


def _soft_recall_precision(gen_descs: list, human_descs: list) -> tuple:
    """
    For each human desc, max similarity to any generated desc -> mean = soft recall.
    For each generated desc, max similarity to any human desc -> mean = soft precision.
    """
    if not human_descs:
        return 0.0, 0.0
    if not gen_descs:
        return 0.0, 0.0
    human_norm = [_normalize_for_compare(d) for d in human_descs]
    gen_norm = [_normalize_for_compare(d) for d in gen_descs]
    recall_scores = []
    for h in human_norm:
        best = max((_text_similarity(h, g) for g in gen_norm), default=0.0)
        recall_scores.append(best)
    precision_scores = []
    for g in gen_norm:
        best = max((_text_similarity(g, h) for h in human_norm), default=0.0)
        precision_scores.append(best)
    soft_recall = sum(recall_scores) / len(recall_scores)
    soft_precision = sum(precision_scores) / len(precision_scores)
    return soft_recall, soft_precision


def _distribution_overlap(gen_series, human_series) -> float:
    """Jaccard overlap of value distributions (normalized counts)."""
    g_vals = gen_series.dropna().astype(str).str.strip()
    h_vals = human_series.dropna().astype(str).str.strip()
    g_vals = g_vals[g_vals != ""]
    h_vals = h_vals[h_vals != ""]
    if g_vals.empty and h_vals.empty:
        return 1.0
    g_set = set(g_vals.str.lower())
    h_set = set(h_vals.str.lower())
    if not g_set and not h_set:
        return 1.0
    if not g_set or not h_set:
        return 0.0
    return len(g_set & h_set) / len(g_set | h_set)


def compute_bulk_metrics(df_gen: pd.DataFrame, df_human: pd.DataFrame) -> dict:
    """Compute bulk-level performance metrics (no row matching)."""
    n_gen = len(df_gen)
    n_human = len(df_human)
    count_ratio = n_gen / n_human if n_human else 0.0
    count_diff = n_gen - n_human

    # Field completeness (% non-empty) for each corpus
    fields = ["Name", "Page Number", "Section", "Description", "VerificationMethod", "Tags", "RequirementType"]
    completeness_gen = {f: _field_completeness(df_gen, f) for f in fields}
    completeness_human = {f: _field_completeness(df_human, f) for f in fields}

    # Average description length
    desc_col = "Description"
    avg_len_gen = df_gen[desc_col].apply(lambda x: len(_normalize_for_compare(x))).mean() if desc_col in df_gen.columns and n_gen else 0
    avg_len_human = df_human[desc_col].apply(lambda x: len(_normalize_for_compare(x))).mean() if desc_col in df_human.columns and n_human else 0

    # Soft recall/precision (description similarity without ID matching)
    gen_descs = df_gen[desc_col].tolist() if desc_col in df_gen.columns else []
    human_descs = df_human[desc_col].tolist() if desc_col in df_human.columns else []
    soft_recall, soft_precision = _soft_recall_precision(gen_descs, human_descs)
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0

    # Lexical overlap: Jaccard of unique words across all descriptions
    all_words_gen = set()
    for d in gen_descs:
        all_words_gen |= _word_set(d)
    all_words_human = set()
    for d in human_descs:
        all_words_human |= _word_set(d)
    if all_words_gen or all_words_human:
        lexical_jaccard = len(all_words_gen & all_words_human) / len(all_words_gen | all_words_human)
    else:
        lexical_jaccard = 1.0

    # Distribution overlap for VerificationMethod, RequirementType
    verif_overlap = 0.0
    type_overlap = 0.0
    if "VerificationMethod" in df_gen.columns and "VerificationMethod" in df_human.columns:
        verif_overlap = _distribution_overlap(df_gen["VerificationMethod"], df_human["VerificationMethod"])
    if "RequirementType" in df_gen.columns and "RequirementType" in df_human.columns:
        type_overlap = _distribution_overlap(df_gen["RequirementType"], df_human["RequirementType"])

    # Tags: aggregate all tags, Jaccard of tag vocabularies
    tags_gen = set()
    if "Tags" in df_gen.columns:
        for t in df_gen["Tags"]:
            tags_gen |= _tags_to_set(t)
    tags_human = set()
    if "Tags" in df_human.columns:
        for t in df_human["Tags"]:
            tags_human |= _tags_to_set(t)
    tags_jaccard = len(tags_gen & tags_human) / len(tags_gen | tags_human) if (tags_gen or tags_human) else 1.0

    return {
        "n_generated": n_gen,
        "n_human": n_human,
        "count_ratio": count_ratio,
        "count_diff": count_diff,
        "completeness_gen": completeness_gen,
        "completeness_human": completeness_human,
        "avg_desc_len_gen": avg_len_gen,
        "avg_desc_len_human": avg_len_human,
        "soft_recall": soft_recall,
        "soft_precision": soft_precision,
        "soft_f1": soft_f1,
        "lexical_jaccard": lexical_jaccard,
        "verification_method_overlap": verif_overlap,
        "requirement_type_overlap": type_overlap,
        "tags_vocab_jaccard": tags_jaccard,
    }


def create_metrics_explanation_df() -> pd.DataFrame:
    """Create a sheet explaining each bulk metric."""
    rows = [
        {"Metric": "Count Ratio", "Description": "Generated count / Human count.", "Interpretation": "~1.0 = similar extraction volume; >1 = AI extracted more; <1 = AI extracted fewer."},
        {"Metric": "Count Difference", "Description": "Generated count - Human count.", "Interpretation": "Positive = AI extracted more requirements; negative = AI missed some."},
        {"Metric": "Soft Recall", "Description": "For each human requirement, max similarity to any generated. Mean across human.", "Interpretation": "Do human requirements have a similar counterpart in AI output? High = few misses."},
        {"Metric": "Soft Precision", "Description": "For each generated requirement, max similarity to any human. Mean across generated.", "Interpretation": "Do generated requirements resemble human ones? High = fewer spurious extractions."},
        {"Metric": "Soft F1", "Description": "Harmonic mean of Soft Recall and Soft Precision.", "Interpretation": "Balanced bulk extraction quality."},
        {"Metric": "Lexical Jaccard", "Description": "Jaccard similarity of unique words across all descriptions (generated vs human).", "Interpretation": "Vocabulary overlap; high = similar terminology used."},
        {"Metric": "VerificationMethod Overlap", "Description": "Jaccard of distinct VerificationMethod values (Test, Inspection, etc.) in each corpus.", "Interpretation": "Do both use similar verification categories?"},
        {"Metric": "RequirementType Overlap", "Description": "Jaccard of distinct RequirementType values (Functional, Interface, etc.).", "Interpretation": "Do both classify into similar types?"},
        {"Metric": "Tags Vocab Jaccard", "Description": "Jaccard of aggregate tag sets (TBD, Updated, etc.) across all rows.", "Interpretation": "Do both identify similar tag types?"},
        {"Metric": "Field Completeness", "Description": "% of rows with non-empty value per field, computed separately for each corpus.", "Interpretation": "Higher = less missing data; compare generated vs human."},
        {"Metric": "Avg Description Length", "Description": "Mean character count of Description field per corpus.", "Interpretation": "Similar lengths suggest similar extraction granularity."},
    ]
    return pd.DataFrame(rows)


# =============================================================================
# Verbatim source-text matching
# =============================================================================
# Goal: for each AI-generated requirement, find the contiguous span from the
# original input document that most closely matches the requirement's
# Description, so the user can audit / quote the source verbatim.
#
# Strategy:
#   1. Extract text from the source document, preserving page numbers when
#      possible (PDF only).
#   2. Split each page's text into sentences (long enough to be meaningful).
#   3. For each generated requirement:
#      a. Restrict candidates to the requirement's Page Number when present
#         (with full-document fallback if nothing matches there).
#      b. Use a cheap word-overlap pre-filter to keep the top-K candidates.
#      c. Score 1..N-sentence sliding windows over those candidates with
#         difflib.SequenceMatcher; return the highest-scoring span.

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Matches table-of-contents-style "dotted leader" lines, e.g.
#   "3.3.5.2.7 Ballistic targets ............................................ 42"
# These are not real prose and should not be used as verbatim source.
_TOC_LINE_RE = re.compile(r"\.{4,}\s*\d{1,4}\s*$")
# Minimum word count for a sentence to be indexed (filters out fragments like
# page numbers, isolated tokens, etc.).
_MIN_SENTENCE_WORDS = 4
# Maximum span (in sentences) considered when assembling a verbatim match.
_DEFAULT_MAX_WINDOW = 3
# Below this similarity, no match is reported (avoids returning random text).
_DEFAULT_MIN_RATIO = 0.25
# When a page-restricted search scores below this, fall back to whole-document
# search (the AI-assigned page number is often wrong for boilerplate pages).
_PAGE_FALLBACK_RATIO = 0.45
# Word-overlap pre-filter: keep at most this many candidate sentences per
# requirement before running the (more expensive) SequenceMatcher.
_PREFILTER_TOPK = 40
# Content-word overlap floor (Jaccard) between description and matched span.
# Filters out spurious character-level matches dominated by shared boilerplate
# like "The system shall ...".
_CONTENT_OVERLAP_MIN = 0.15
# Stopwords excluded from the content-overlap check (very common spec
# vocabulary that doesn't carry topical meaning).
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
    "is", "are", "be", "by", "as", "at", "it", "its", "this", "that", "these",
    "those", "from", "into", "shall", "must", "should", "will", "may", "can",
    "no", "not", "all", "any", "each", "such", "via", "out", "if", "than",
    "then", "when", "where", "while", "per", "without", "within", "between",
    "among", "over", "under", "above", "below", "after", "before", "during",
    "but", "do", "does", "did", "has", "have", "had", "been", "being", "their",
    "they", "them", "his", "her", "him", "she", "he", "we", "us", "our", "you",
    "your", "i", "me", "my", "one", "two", "etc",
})


def _content_words(text: str) -> set:
    """Lower-cased content words (non-stopwords) from text."""
    return {w for w in (m.lower() for m in _WORD_RE.findall(text)) if w not in _STOPWORDS}


def _extract_source_pages(path: str) -> list[tuple[int | None, str]]:
    """
    Extract text from the source document as (page_number, page_text) tuples.

    Page numbers are 1-based for PDFs; for DOCX/TXT we return a single
    (None, full_text) tuple because no reliable page mapping is available.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source document not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber  # local import to keep base script light
        except ImportError as e:
            raise ImportError(
                "Reading the source PDF requires pdfplumber. "
                "Install with: pip install pdfplumber"
            ) from e
        pages: list[tuple[int | None, str]] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text))
        return pages
    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError as e:
            raise ImportError(
                "Reading the source DOCX requires python-docx. "
                "Install with: pip install python-docx"
            ) from e
        d = docx.Document(path)
        full = "\n".join(par.text for par in d.paragraphs)
        return [(None, full)]
    if suffix in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [(None, f.read())]
    raise ValueError(
        f"Unsupported source document type '{suffix}'. "
        "Supported: .pdf, .docx, .txt, .md"
    )


def _split_sentences(text: str) -> list[str]:
    """Split a block of text into sentence-like fragments, dropping TOC/noise lines."""
    if not text:
        return []
    out: list[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(text):
        s = raw.strip()
        if not s:
            continue
        if len(_WORD_RE.findall(s)) < _MIN_SENTENCE_WORDS:
            continue
        # Drop dotted-leader TOC lines like "3.1.2 Section ........... 21".
        if _TOC_LINE_RE.search(s):
            continue
        # Collapse internal dotted-leader runs that survived split (e.g. multi-
        # line TOC blocks fused into one "sentence").
        s = re.sub(r"\.{4,}", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(_WORD_RE.findall(s)) < _MIN_SENTENCE_WORDS:
            continue
        out.append(s)
    return out


def _build_source_index(
    pages: list[tuple[int | None, str]],
) -> list[tuple[int | None, str, set]]:
    """
    Flatten pages into a list of (page_number, sentence, word_set) tuples.
    Pre-computing word_set per sentence enables a fast overlap pre-filter.
    """
    index: list[tuple[int | None, str, set]] = []
    for page_num, text in pages:
        for sent in _split_sentences(text):
            words = {w.lower() for w in _WORD_RE.findall(sent)}
            index.append((page_num, sent, words))
    return index


def _parse_page_number(val) -> int | None:
    """Coerce a Page Number cell to an int (handles '1', '1.0', '1-2', etc.)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


def _find_verbatim_source(
    description: str,
    source_index: list[tuple[int | None, str, set]],
    page_filter: int | None = None,
    max_window: int = _DEFAULT_MAX_WINDOW,
    min_ratio: float = _DEFAULT_MIN_RATIO,
) -> tuple[str, float, int | None]:
    """
    Find the contiguous 1..max_window-sentence span in source_index whose text
    most closely matches `description`.

    Returns (verbatim_text, similarity_ratio, page_number_of_match). When no
    span scores above `min_ratio`, returns ("", best_ratio, None).
    """
    desc = (description or "").strip()
    if not desc or not source_index:
        return "", 0.0, None

    desc_words = {w.lower() for w in _WORD_RE.findall(desc)}
    if not desc_words:
        return "", 0.0, None

    # Build candidate index list, optionally restricted by page.
    if page_filter is not None:
        page_candidates = [i for i, (p, _, _) in enumerate(source_index) if p == page_filter]
        if not page_candidates:
            # Page filter found nothing -> fall back to whole document
            page_candidates = list(range(len(source_index)))
    else:
        page_candidates = list(range(len(source_index)))

    # Cheap word-overlap pre-filter: rank candidates by |desc ∩ sent| / |desc|.
    scored = []
    for idx in page_candidates:
        _, _, words = source_index[idx]
        if not words:
            continue
        overlap = len(desc_words & words)
        if overlap == 0:
            continue
        scored.append((overlap, idx))
    if not scored:
        return "", 0.0, None
    scored.sort(reverse=True)
    top_indices = [idx for _, idx in scored[:_PREFILTER_TOPK]]

    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(desc)

    n = len(source_index)
    best_text = ""
    best_ratio = 0.0
    best_page: int | None = None

    for start in top_indices:
        for w in range(1, max_window + 1):
            end = start + w
            if end > n:
                break
            span = " ".join(source_index[start + k][1] for k in range(w))
            matcher.set_seq1(span)
            # Two cheap upper-bound checks before computing the real ratio.
            if matcher.real_quick_ratio() < best_ratio:
                continue
            if matcher.quick_ratio() < best_ratio:
                continue
            r = matcher.ratio()
            if r > best_ratio:
                best_ratio = r
                best_text = span
                best_page = source_index[start][0]

    if best_ratio < min_ratio:
        return "", 0.0, None
    # Spurious-match guard: require non-trivial topical overlap (content words)
    # between the description and the matched span. Without this, SequenceMatcher
    # can latch onto shared boilerplate like "The widget shall ...".
    desc_content = _content_words(desc)
    span_content = _content_words(best_text)
    union = desc_content | span_content
    overlap_jaccard = (len(desc_content & span_content) / len(union)) if union else 0.0
    if overlap_jaccard < _CONTENT_OVERLAP_MIN:
        return "", 0.0, None
    return best_text, best_ratio, best_page


def build_verbatim_source_df(df_gen: pd.DataFrame, source_path: str) -> pd.DataFrame:
    """
    Return a DataFrame that mirrors df_gen and appends two new columns to the
    main output table:
        - "Source Text (Verbatim)": best-matching contiguous span from the
          original input document for this requirement's Description.
        - "Source Match Score": similarity in [0, 1] of that span vs the
          Description (1.0 = exact). Empty verbatim cells indicate that no
          span scored above the acceptance threshold (a 1:1 mapping is not
          guaranteed).
    """
    pages = _extract_source_pages(source_path)
    source_index = _build_source_index(pages)
    if not source_index:
        raise ValueError(
            f"No usable text could be extracted from source document: {source_path}"
        )

    desc_col = "Description" if "Description" in df_gen.columns else None
    page_col = "Page Number" if "Page Number" in df_gen.columns else None

    verbatim_texts: list[str] = []
    scores: list[float] = []

    for _, row in df_gen.iterrows():
        description = row.get(desc_col, "") if desc_col else ""
        desc_str = str(description) if description is not None else ""
        page_filter = _parse_page_number(row.get(page_col)) if page_col else None
        text, ratio, _ = _find_verbatim_source(
            description=desc_str,
            source_index=source_index,
            page_filter=page_filter,
        )
        # When the page-restricted match is weak, retry without the page filter.
        # AI-assigned page numbers are sometimes wrong (e.g. everything tagged
        # page 1 when the content actually lives later in the document).
        if page_filter is not None and ratio < _PAGE_FALLBACK_RATIO:
            alt_text, alt_ratio, _ = _find_verbatim_source(
                description=desc_str,
                source_index=source_index,
                page_filter=None,
            )
            if alt_ratio > ratio:
                text, ratio = alt_text, alt_ratio
        verbatim_texts.append(text)
        scores.append(round(ratio, 4))

    out = df_gen.copy()
    out["Source Text (Verbatim)"] = verbatim_texts
    out["Source Match Score"] = scores
    return out


def _derive_enriched_path(generated_path: str, override: str | None) -> str:
    """Pick an output path for the enriched (with-verbatim) requirements Excel."""
    if override:
        return override
    stem = Path(generated_path).stem or "generated_requirements"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_with_verbatim_{ts}.xlsx"


def run_comparison(
    generated_path: str,
    human_path: str,
    output_path: str | None = None,
    source_doc_path: str | None = None,
    enriched_output_path: str | None = None,
) -> tuple[str, str | None]:
    """
    Load both files, compute bulk metrics, write the metrics Excel and -- when
    a source document is provided -- a separate enriched generated-requirements
    Excel with the verbatim source columns appended.

    Returns (metrics_output_path, enriched_output_path_or_None).
    """
    df_gen = load_requirements(generated_path)
    df_human = load_requirements(human_path)
    m = compute_bulk_metrics(df_gen, df_human)

    if output_path is None:
        output_path = f"compare_requirements_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary sheet
        summary_data = [
            ["Metric", "Value", "Interpretation"],
            ["Total requirements (AI-generated)", m["n_generated"], "Count extracted by AI"],
            ["Total requirements (Human reference)", m["n_human"], "Count in human reference"],
            ["Count Ratio (gen/human)", f"{m['count_ratio']:.4f}", "~1 = similar volume"],
            ["Count Difference (gen - human)", m["count_diff"], ""],
            ["", "", ""],
            ["Soft Recall", f"{m['soft_recall']:.4f}", "Human reqs with similar AI counterpart"],
            ["Soft Precision", f"{m['soft_precision']:.4f}", "Generated reqs similar to human"],
            ["Soft F1", f"{m['soft_f1']:.4f}", "Balanced bulk quality"],
            ["", "", ""],
            ["Lexical Jaccard (word overlap)", f"{m['lexical_jaccard']:.4f}", "Vocabulary overlap"],
            ["VerificationMethod Overlap", f"{m['verification_method_overlap']:.4f}", ""],
            ["RequirementType Overlap", f"{m['requirement_type_overlap']:.4f}", ""],
            ["Tags Vocab Jaccard", f"{m['tags_vocab_jaccard']:.4f}", ""],
            ["", "", ""],
            ["Avg Description Length (generated)", f"{m['avg_desc_len_gen']:.1f}", "chars"],
            ["Avg Description Length (human)", f"{m['avg_desc_len_human']:.1f}", "chars"],
        ]
        pd.DataFrame(summary_data).to_excel(writer, index=False, header=False, sheet_name="Summary")

        # Field completeness comparison
        comp_rows = []
        for f in ["Name", "Page Number", "Section", "Description", "VerificationMethod", "Tags", "RequirementType"]:
            comp_rows.append({
                "Field": f,
                "Generated % Non-Empty": f"{m['completeness_gen'].get(f, 0) * 100:.1f}",
                "Human % Non-Empty": f"{m['completeness_human'].get(f, 0) * 100:.1f}",
            })
        pd.DataFrame(comp_rows).to_excel(writer, index=False, sheet_name="Field_Completeness")

        # Distributions (value counts for each corpus)
        for col in ["VerificationMethod", "RequirementType"]:
            if col in df_gen.columns and col in df_human.columns:
                g_counts = df_gen[col].fillna("").astype(str).str.strip()
                g_counts = g_counts[g_counts != ""].value_counts()
                h_counts = df_human[col].fillna("").astype(str).str.strip()
                h_counts = h_counts[h_counts != ""].value_counts()
                dist_df = pd.DataFrame({
                    "Generated_Count": g_counts,
                    "Human_Count": h_counts,
                }).fillna(0).astype(int)
                dist_df.to_excel(writer, sheet_name=f"Dist_{col[:10]}")

        # Metrics explanation
        create_metrics_explanation_df().to_excel(writer, index=False, sheet_name="Metrics_Explanation")

    enriched_path = None
    if source_doc_path:
        enriched_path = _derive_enriched_path(generated_path, enriched_output_path)
        verbatim_df = build_verbatim_source_df(df_gen, source_doc_path)
        with pd.ExcelWriter(enriched_path, engine="openpyxl") as writer:
            verbatim_df.to_excel(writer, index=False, sheet_name="Requirements")

    return output_path, enriched_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare bulk of AI-generated vs human-made requirement Excel outputs.",
    )
    parser.add_argument("generated", nargs="?", default=GENERATED_EXCEL_PATH, help="Path to AI-generated Excel")
    parser.add_argument("human", nargs="?", default=HUMAN_EXCEL_PATH, help="Path to human-made Excel")
    parser.add_argument("-o", "--output", default=OUTPUT_EXCEL_PATH, help="Output Excel path")
    parser.add_argument(
        "-s",
        "--source",
        default=SOURCE_DOCUMENT_PATH,
        help=(
            "Optional path to the ORIGINAL input document (.pdf, .docx, .txt) "
            "that the AI extracted requirements from. When provided, a second "
            "Excel file is produced: a copy of the generated requirements "
            "table with two extra columns ('Source Text (Verbatim)', "
            "'Source Match Score') appended to the main table."
        ),
    )
    parser.add_argument(
        "--enriched-output",
        default=ENRICHED_OUTPUT_PATH,
        help=(
            "Optional path for the enriched generated-requirements Excel "
            "(only used with --source). Defaults to "
            "'<generated-stem>_with_verbatim_<timestamp>.xlsx'."
        ),
    )
    args = parser.parse_args()

    if "path/to" in str(args.generated) or "path/to" in str(args.human):
        print("ERROR: Set GENERATED_EXCEL_PATH and HUMAN_EXCEL_PATH, or pass as arguments:")
        print("       python compare_requirements_performance.py <generated.xlsx> <human.xlsx>")
        sys.exit(1)

    # Source document is optional; only use it if a real path was given.
    source_doc = args.source
    if not source_doc or "path/to" in str(source_doc):
        source_doc = None

    try:
        metrics_out, enriched_out = run_comparison(
            args.generated,
            args.human,
            args.output,
            source_doc_path=source_doc,
            enriched_output_path=args.enriched_output,
        )
        print(f"Metrics written to: {metrics_out}")
        if enriched_out:
            print(f"Enriched generated requirements (with verbatim source) written to: {enriched_out}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except BadZipFile as e:
        print(
            "Error: One of the files is not a valid .xlsx. Ensure both files are saved as "
            ".xlsx (Excel 2007+). If you have .xls or CSV, save/re-save as .xlsx from Excel."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
