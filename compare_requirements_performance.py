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
    An Excel file (compare_requirements_metrics_<timestamp>.xlsx) containing:
    - Summary: Bulk performance metrics
    - Distributions: VerificationMethod, RequirementType, Tags comparison
    - Metrics_Explanation: Description of each metric and what it measures
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


def run_comparison(generated_path: str, human_path: str, output_path: str | None = None) -> str:
    """Load both files, compute bulk metrics, write output Excel."""
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

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare bulk of AI-generated vs human-made requirement Excel outputs.",
    )
    parser.add_argument("generated", nargs="?", default=GENERATED_EXCEL_PATH, help="Path to AI-generated Excel")
    parser.add_argument("human", nargs="?", default=HUMAN_EXCEL_PATH, help="Path to human-made Excel")
    parser.add_argument("-o", "--output", default=OUTPUT_EXCEL_PATH, help="Output Excel path")
    args = parser.parse_args()

    if "path/to" in str(args.generated) or "path/to" in str(args.human):
        print("ERROR: Set GENERATED_EXCEL_PATH and HUMAN_EXCEL_PATH, or pass as arguments:")
        print("       python compare_requirements_performance.py <generated.xlsx> <human.xlsx>")
        sys.exit(1)

    try:
        out = run_comparison(args.generated, args.human, args.output)
        print(f"Metrics written to: {out}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
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
