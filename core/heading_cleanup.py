"""
heading_cleanup.py
------------------
PDF heading correction for the Docling ingestion pipeline. Combines three
concerns into one module:

  1. ToC extraction        — pull the real heading list from a PDF's clickable
                             table-of-contents link annotations, and normalize
                             headings for comparison.
  2. Heading-node cleanup  — remove page headers/footers and false SECTION_HEADER
                             nodes from a DoclingDocument before chunking.
  3. Parent reconstruction — expand a chunk's single leaf heading into its full
                             ancestor chain using the ToC as ground truth.

Typical use inside chunk_documents() on the PDF path:

    dl_doc = converter.convert(source=pdf_path).document
    toc_lines = extract_toc_lines(pdf_path)
    toc_set   = build_toc_set(toc_lines)
    dl_doc, _ = clean_heading_nodes(dl_doc, toc_set)
    toc_index = build_toc_index(toc_lines)
    # ... chunk dl_doc, then per chunk:
    expanded  = expand_headings_with_parents(c.meta.headings, toc_index)

Requires PyMuPDF (imported as `fitz`) for ToC extraction. The unrelated PyPI
package also named `fitz` will shadow PyMuPDF if installed — ensure `pymupdf`
is the one present. ToC extraction degrades gracefully (returns []) if PyMuPDF
is missing, in which case headings fall back to Docling's raw output.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict, Tuple

from docling_core.types.doc import DocItemLabel

from .logger_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared regexes
# ---------------------------------------------------------------------------

# ToC entries for figures/tables, e.g. "Figure 3-1", "Table 4.2" — excluded.
_FIG_TABLE_RE = re.compile(r"^\s*(figure|table)\s+[\d\.\-]+", re.IGNORECASE)

# Leading section number: "1", "1.1", "3.4.5", optional trailing dot.
# Paren-style ("4) Foo") is intentionally NOT treated as a strippable section
# number, so it won't collapse to a bare word that false-matches the ToC.
_LEADING_NUM_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+")

# Trailing dotted leader + page number, e.g. "Scope ........ 14".
_DOTTED_LEADER_RE = re.compile(r"\s*\.{2,}\s*\d+\s*$")

# A heading "is numbered" if it starts with 1, 1.2, 3.4.5 (optional trailing
# dot). Dot-only, no paren — consistent with normalization above.
_NUM_PREFIX_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s")

# Leading section number capture for parent reconstruction: "2.1.2" etc.
_NUM_CAPTURE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


# ---------------------------------------------------------------------------
# 1. ToC extraction & normalization
# ---------------------------------------------------------------------------

def normalize_heading(s: str) -> str:
    """
    Normalize a heading or ToC line for comparison: drop trailing dotted leader
    + page number, strip the leading section number, lowercase, collapse
    whitespace. Applied identically to both ToC lines and detected headings so
    the two compare consistently.
    """
    s = s.strip()
    s = _DOTTED_LEADER_RE.sub("", s)
    s = _LEADING_NUM_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def extract_toc_lines(pdf_path: str, max_pages: int = 12) -> List[str]:
    """
    Return the ToC heading lines from a PDF, in document order, by reading the
    GoTo link annotations on the first `max_pages` pages. Figure/Table entries
    are excluded. Returns [] if PyMuPDF is unavailable or no links are found.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) not available; cannot extract PDF ToC. "
            "Install `pymupdf` to enable ToC-based heading correction."
        )
        return []

    lines: List[str] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Could not open PDF '%s' for ToC extraction: %s", pdf_path, exc)
        return []

    try:
        for page_num in range(min(max_pages, doc.page_count)):
            page = doc[page_num]
            for link in page.get_links():
                if link.get("kind") != fitz.LINK_GOTO:
                    continue
                text = page.get_textbox(link["from"]).strip()
                if not text:
                    continue
                if _FIG_TABLE_RE.match(text):
                    continue
                lines.append(text)
    finally:
        doc.close()

    logger.info("Extracted %d ToC heading line(s) from '%s'.", len(lines), pdf_path)
    return lines


def build_toc_set(toc_lines: List[str]) -> set:
    """
    Normalized exact-match lookup set for ToC membership tests.
    Use this (not substring matching against a joined string) to decide whether
    a detected heading appears in the ToC.
    """
    return {normalize_heading(line) for line in toc_lines if line.strip()}


# ---------------------------------------------------------------------------
# 2. Heading-node cleanup
# ---------------------------------------------------------------------------

_EXCLUDE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


def is_numbered_heading(text: str) -> bool:
    return bool(_NUM_PREFIX_RE.match(text))


def clean_heading_nodes(
    dl_doc,
    toc_set: set | None = None,
    repeated_threshold: int = 3,
) -> Tuple[object, dict]:
    """
    Remove header/footer and false SECTION_HEADER nodes from `dl_doc` in place.

    Rules:
      - Remove PAGE_HEADER / PAGE_FOOTER items (Docling's own classification).
      - Remove SECTION_HEADER items whose exact text repeats >= repeated_threshold
        times (running headers mislabeled as headings).
      - Remove SECTION_HEADER items that are neither numbered nor present in the
        ToC ground truth (false headers). A heading is kept if it is numbered OR
        in the ToC.

    Args:
        dl_doc: a DoclingDocument (from converter.convert(...).document).
        toc_set: normalized ToC membership set (from build_toc_set). If None or
            empty, the ToC test is skipped and only the numbered/label/repeat
            rules apply.
        repeated_threshold: repeat count at/above which a SECTION_HEADER is
            treated as a running header.

    Returns:
        (dl_doc, stats) where stats reports counts removed by reason.
    """
    toc_set = toc_set or set()

    # 1. Find repeated SECTION_HEADER texts (running headers).
    header_text_counts: Counter = Counter()
    for item, _ in dl_doc.iterate_items():
        if item.label == DocItemLabel.SECTION_HEADER and item.text.strip():
            header_text_counts[item.text.strip()] += 1
    repeated = {t for t, c in header_text_counts.items() if c >= repeated_threshold}

    # 2. Collect everything to remove (collect first, delete after — never
    #    mutate the tree mid-iteration).
    to_remove = []
    stats = {"page_header_footer": 0, "repeated_header": 0, "false_header": 0}

    for item, _ in dl_doc.iterate_items():
        if item.label in _EXCLUDE_LABELS:
            to_remove.append(item)
            stats["page_header_footer"] += 1
            continue

        if item.label != DocItemLabel.SECTION_HEADER:
            continue

        text = item.text.strip()

        if text in repeated:
            to_remove.append(item)
            stats["repeated_header"] += 1
            continue

        numbered = is_numbered_heading(text)
        in_toc = normalize_heading(text) in toc_set if toc_set else False

        if not numbered and not in_toc:
            to_remove.append(item)
            stats["false_header"] += 1

    # 3. Delete.
    for item in to_remove:
        dl_doc.delete_items(node_items=[item])

    stats["total_removed"] = len(to_remove)
    logger.info(
        "Heading cleanup: removed %d (page hdr/ftr=%d, repeated=%d, false=%d).",
        stats["total_removed"],
        stats["page_header_footer"],
        stats["repeated_header"],
        stats["false_header"],
    )
    return dl_doc, stats


# ---------------------------------------------------------------------------
# 3. Parent reconstruction from the ToC
# ---------------------------------------------------------------------------

def _section_number(heading: str):
    """'2.1.2 States and Modes' -> '2.1.2' (None if no leading number)."""
    m = _NUM_CAPTURE_RE.match(heading)
    return m.group(1) if m else None


def build_toc_index(toc_headings: List[str]) -> Dict[str, str]:
    """
    Map section-number string -> full heading text, for O(1) parent lookup.
    If the same number appears twice, the last occurrence wins.
    """
    index: Dict[str, str] = {}
    for h in toc_headings:
        num = _section_number(h)
        if num:
            index[num] = h.strip()
    return index


def _ancestor_numbers(num: str) -> List[str]:
    """'2.1.2' -> ['2', '2.1', '2.1.2'] (each prefix, root->leaf)."""
    parts = num.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def expand_headings_with_parents(chunk_headings, toc_index: Dict[str, str]):
    """
    Expand a chunk's headings list (typically one numbered leaf) into the full
    ancestor chain using the ToC index.

    - Unnumbered or unknown leaves are returned unchanged.
    - Missing intermediate parents fall back to the bare section number so depth
      (and therefore indentation) is preserved rather than silently dropped.
    """
    if not chunk_headings:
        return chunk_headings

    leaf = chunk_headings[-1].strip()
    leaf_num = _section_number(leaf)
    if not leaf_num:
        return list(chunk_headings)

    chain = []
    for anc_num in _ancestor_numbers(leaf_num):
        if anc_num == leaf_num:
            chain.append(toc_index.get(anc_num, leaf))
        else:
            chain.append(toc_index.get(anc_num, anc_num))
    return chain


def headings_before_first_numbered(headings: List[str]) -> List[str]:
    """
    Return the front-matter headings appearing before the first numbered heading
    (e.g. 'Foreword', 'Scope' before '1. Introduction'). Used to skip
    front-matter chunks during requirement generation.
    """
    pre = []
    for h in headings:
        if _NUM_PREFIX_RE.match(h):
            break
        pre.append(h)
    return pre
