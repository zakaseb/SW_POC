#!/usr/bin/env python3
"""
Performance Metrics Comparison: AI-Generated vs Human-Made Requirement Excel Outputs

Compares an AI-generated requirements Excel sheet against a human-created reference
(gold standard) and produces an Excel report with performance metrics.

Usage:
    1. Set GENERATED_EXCEL_PATH and HUMAN_EXCEL_PATH below (or pass as arguments)
    2. Run: python compare_requirements_performance.py
           or: python compare_requirements_performance.py <generated.xlsx> <human.xlsx>

Output:
    An Excel file (compare_requirements_metrics_<timestamp>.xlsx) containing:
    - Summary: Overall performance metrics
    - Per-Requirement: Field-by-field comparison for matched requirements
    - Metrics_Explanation: Description of each metric and what it measures
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# =============================================================================
# CONFIGURATION: Set your file paths here
# =============================================================================
# Path to the AI-generated requirements Excel output
GENERATED_EXCEL_PATH = "path/to/generated_requirements.xlsx"

# Path to the human-made (reference/gold standard) requirements Excel
HUMAN_EXCEL_PATH = "path/to/human_requirements.xlsx"

# Path for the output metrics Excel (optional; default: timestamped in current dir)
OUTPUT_EXCEL_PATH = None  # e.g., "document_store/performance_metrics.xlsx"

# =============================================================================

EXPECTED_COLUMNS = [
    "Name",
    "Page Number",
    "Section",
    "Description",
    "VerificationMethod",
    "Tags",
    "RequirementType",
    "DocumentRequirementID",
]


def _normalize_id(val) -> str:
    """Normalize requirement ID for matching (strip whitespace, lowercase)."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def _normalize_for_compare(val) -> str:
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


def _tags_jaccard(tags_a: str, tags_b: str) -> float:
    """Jaccard similarity of tag sets (comma- or space-separated)."""
    def to_set(s):
        if pd.isna(s) or not str(s).strip():
            return set()
        parts = re.split(r"[,;\s]+", str(s).strip())
        return {p.strip().lower() for p in parts if p.strip()}

    sa, sb = to_set(tags_a), to_set(tags_b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _exact_match(a: str, b: str) -> bool:
    """Check if two normalized values match exactly."""
    na, nb = _normalize_for_compare(a), _normalize_for_compare(b)
    return na == nb


def load_requirements(path: str) -> pd.DataFrame:
    """Load requirements from an Excel file. Expects first sheet to contain requirements."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path, engine="openpyxl", sheet_name=0)
    # Normalize column names (strip, handle variations)
    df.columns = [str(c).strip() for c in df.columns]
    # Map common alternates
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


def compute_metrics(df_gen: pd.DataFrame, df_human: pd.DataFrame) -> dict:
    """
    Compute performance metrics by matching requirements on DocumentRequirementID.
    """
    id_col = "DocumentRequirementID"
    if id_col not in df_gen.columns or id_col not in df_human.columns:
        raise ValueError(
            "Both sheets must have a 'DocumentRequirementID' column for matching. "
            f"Generated: {list(df_gen.columns)}; Human: {list(df_human.columns)}"
        )

    # Build lookup by ID
    human_by_id = {}
    for _, row in df_human.iterrows():
        rid = _normalize_id(row.get(id_col, ""))
        if rid:
            human_by_id[rid] = row

    gen_by_id = {}
    for _, row in df_gen.iterrows():
        rid = _normalize_id(row.get(id_col, ""))
        if rid:
            gen_by_id[rid] = row

    # Match on ID
    common_ids = set(gen_by_id.keys()) & set(human_by_id.keys())
    only_gen = set(gen_by_id.keys()) - set(human_by_id.keys())
    only_human = set(human_by_id.keys()) - set(gen_by_id.keys())

    total_gen = len(gen_by_id) if gen_by_id else 0
    total_human = len(human_by_id) if human_by_id else 0
    matched = len(common_ids)

    # Precision: of what we generated, how many are in the reference?
    precision = matched / total_gen if total_gen else 0.0
    # Recall: of what the human extracted, how many did we get?
    recall = matched / total_human if total_human else 0.0
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Field-level accuracy for matched pairs
    field_acc = {
        "Name": [],
        "Description": [],
        "VerificationMethod": [],
        "Tags": [],
        "RequirementType": [],
        "Page Number": [],
        "Section": [],
    }

    details_rows = []

    for rid in sorted(common_ids):
        r_gen = gen_by_id[rid]
        r_human = human_by_id[rid]

        # Name: text similarity
        name_sim = _text_similarity(
            _normalize_for_compare(r_gen.get("Name", "")),
            _normalize_for_compare(r_human.get("Name", "")),
        )
        field_acc["Name"].append(name_sim)

        # Description: text similarity
        desc_sim = _text_similarity(
            _normalize_for_compare(r_gen.get("Description", "")),
            _normalize_for_compare(r_human.get("Description", "")),
        )
        field_acc["Description"].append(desc_sim)

        # VerificationMethod, RequirementType: exact match
        verif_match = _exact_match(
            r_gen.get("VerificationMethod", ""),
            r_human.get("VerificationMethod", ""),
        )
        type_match = _exact_match(
            r_gen.get("RequirementType", ""),
            r_human.get("RequirementType", ""),
        )
        field_acc["VerificationMethod"].append(1.0 if verif_match else 0.0)
        field_acc["RequirementType"].append(1.0 if type_match else 0.0)

        # Tags: Jaccard
        tags_sim = _tags_jaccard(
            r_gen.get("Tags", ""),
            r_human.get("Tags", ""),
        )
        field_acc["Tags"].append(tags_sim)

        # Page Number, Section: exact match (normalized)
        pg_match = _exact_match(r_gen.get("Page Number", ""), r_human.get("Page Number", ""))
        sec_match = _exact_match(r_gen.get("Section", ""), r_human.get("Section", ""))
        field_acc["Page Number"].append(1.0 if pg_match else 0.0)
        field_acc["Section"].append(1.0 if sec_match else 0.0)

        details_rows.append({
            "DocumentRequirementID": rid,
            "Name_Similarity": round(name_sim, 4),
            "Description_Similarity": round(desc_sim, 4),
            "VerificationMethod_Match": verif_match,
            "RequirementType_Match": type_match,
            "Tags_Jaccard": round(tags_sim, 4),
            "Page_Number_Match": pg_match,
            "Section_Match": sec_match,
        })

    # Mean field accuracy
    mean_field = {k: sum(v) / len(v) if v else 0.0 for k, v in field_acc.items()}

    return {
        "total_generated": total_gen,
        "total_human": total_human,
        "matched": matched,
        "only_in_generated": len(only_gen),
        "only_in_human": len(only_human),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_field_accuracy": mean_field,
        "details_rows": details_rows,
        "only_gen_ids": sorted(only_gen),
        "only_human_ids": sorted(only_human),
    }


def create_metrics_explanation_df() -> pd.DataFrame:
    """Create a sheet explaining each metric."""
    rows = [
        {
            "Metric": "Precision",
            "Description": "Of all requirements the AI extracted, what fraction exists in the human reference?",
            "Interpretation": "High precision = fewer false positives (AI did not invent requirements).",
        },
        {
            "Metric": "Recall",
            "Description": "Of all requirements in the human reference, what fraction did the AI extract?",
            "Interpretation": "High recall = fewer missed requirements.",
        },
        {
            "Metric": "F1 Score",
            "Description": "Harmonic mean of Precision and Recall.",
            "Interpretation": "Balanced measure of extraction quality; penalizes extreme imbalance.",
        },
        {
            "Metric": "Name_Similarity",
            "Description": "Text similarity (0-1) between AI and human requirement names.",
            "Interpretation": "Measures how closely AI names match human labels.",
        },
        {
            "Metric": "Description_Similarity",
            "Description": "Text similarity (0-1) between AI and human requirement descriptions.",
            "Interpretation": "Core measure of whether the AI captured the requirement text correctly.",
        },
        {
            "Metric": "VerificationMethod_Match",
            "Description": "Exact match (yes/no) for VerificationMethod (Test, Inspection, Analysis).",
            "Interpretation": "Accuracy of verification method classification.",
        },
        {
            "Metric": "RequirementType_Match",
            "Description": "Exact match (yes/no) for RequirementType (Functional, Interface, Constraint).",
            "Interpretation": "Accuracy of requirement type classification.",
        },
        {
            "Metric": "Tags_Jaccard",
            "Description": "Jaccard similarity (0-1) of tag sets between AI and human.",
            "Interpretation": "How well tags like TBD, Updated are identified.",
        },
        {
            "Metric": "Page_Number_Match",
            "Description": "Exact match (yes/no) for page number.",
            "Interpretation": "Traceability: did the AI correctly assign the source page?",
        },
        {
            "Metric": "Section_Match",
            "Description": "Exact match (yes/no) for section number (e.g., 3.1.2).",
            "Interpretation": "Traceability: did the AI correctly assign the source section?",
        },
    ]
    return pd.DataFrame(rows)


def run_comparison(generated_path: str, human_path: str, output_path: str | None = None) -> str:
    """Load both files, compute metrics, write output Excel. Returns path to output file."""
    df_gen = load_requirements(generated_path)
    df_human = load_requirements(human_path)
    metrics = compute_metrics(df_gen, df_human)

    if output_path is None:
        output_path = f"compare_requirements_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary sheet
        summary_data = [
            ["Metric", "Value", "Interpretation"],
            ["Total requirements (AI-generated)", metrics["total_generated"], "Count of requirements extracted by AI"],
            ["Total requirements (Human reference)", metrics["total_human"], "Count in human-made reference"],
            ["Matched (same DocumentRequirementID)", metrics["matched"], "Requirements found in both"],
            ["Only in AI output", metrics["only_in_generated"], "Potential false positives"],
            ["Only in Human reference", metrics["only_in_human"], "Potential misses"],
            ["", "", ""],
            ["Precision", f"{metrics['precision']:.4f}", "matched / total_generated"],
            ["Recall", f"{metrics['recall']:.4f}", "matched / total_human"],
            ["F1 Score", f"{metrics['f1']:.4f}", "2 * P * R / (P + R)"],
            ["", "", ""],
            ["Mean Name Similarity", f"{metrics['mean_field_accuracy']['Name']:.4f}", ""],
            ["Mean Description Similarity", f"{metrics['mean_field_accuracy']['Description']:.4f}", ""],
            ["Mean VerificationMethod Match Rate", f"{metrics['mean_field_accuracy']['VerificationMethod']:.4f}", ""],
            ["Mean RequirementType Match Rate", f"{metrics['mean_field_accuracy']['RequirementType']:.4f}", ""],
            ["Mean Tags Jaccard", f"{metrics['mean_field_accuracy']['Tags']:.4f}", ""],
            ["Mean Page Number Match Rate", f"{metrics['mean_field_accuracy']['Page Number']:.4f}", ""],
            ["Mean Section Match Rate", f"{metrics['mean_field_accuracy']['Section']:.4f}", ""],
        ]
        pd.DataFrame(summary_data).to_excel(writer, index=False, header=False, sheet_name="Summary")

        # Per-requirement details
        if metrics["details_rows"]:
            pd.DataFrame(metrics["details_rows"]).to_excel(
                writer, index=False, sheet_name="Per-Requirement"
            )

        # Unmatched IDs (optional)
        if metrics["only_gen_ids"] or metrics["only_human_ids"]:
            unmt = pd.DataFrame({
                "Only in AI output (DocumentRequirementID)": metrics["only_gen_ids"]
                + [""] * max(0, len(metrics["only_human_ids"]) - len(metrics["only_gen_ids"])),
                "Only in Human reference (DocumentRequirementID)": metrics["only_human_ids"]
                + [""] * max(0, len(metrics["only_gen_ids"]) - len(metrics["only_human_ids"])),
            })
            unmt.to_excel(writer, index=False, sheet_name="Unmatched_IDs")

        # Metrics explanation
        create_metrics_explanation_df().to_excel(writer, index=False, sheet_name="Metrics_Explanation")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare AI-generated vs human-made requirement Excel outputs and produce performance metrics.",
    )
    parser.add_argument(
        "generated",
        nargs="?",
        default=GENERATED_EXCEL_PATH,
        help="Path to AI-generated requirements Excel",
    )
    parser.add_argument(
        "human",
        nargs="?",
        default=HUMAN_EXCEL_PATH,
        help="Path to human-made (reference) requirements Excel",
    )
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_EXCEL_PATH,
        help="Output Excel path (default: timestamped in current dir)",
    )
    args = parser.parse_args()

    if args.generated == GENERATED_EXCEL_PATH and "path/to" in str(args.generated):
        print("ERROR: Please set GENERATED_EXCEL_PATH and HUMAN_EXCEL_PATH at the top of this script,")
        print("       or pass them as arguments:")
        print("       python compare_requirements_performance.py <generated.xlsx> <human.xlsx>")
        sys.exit(1)
    if args.human == HUMAN_EXCEL_PATH and "path/to" in str(args.human):
        print("ERROR: Please set HUMAN_EXCEL_PATH or pass it as second argument.")
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


if __name__ == "__main__":
    main()
