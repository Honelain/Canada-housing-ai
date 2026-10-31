"""Apply approved audit exclusions; leave all retained feature values unchanged."""

from pathlib import Path

import pandas as pd

from src.audit.audit_pipeline import (EXPECTED_COUNTS, flag_approved_removals, flag_review_candidates)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "canada_housing.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_housing.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "cleaning" / "cleaning_summary.txt"
ORIGINAL_COLUMNS = [
    "City", "Province", "Latitude", "Longitude", "Price", "Bedrooms", "Bathrooms",
    "Acreage", "Property Type", "Square Footage", "Garage", "Parking", "Basement",
    "Exterior", "Fireplace", "Heating", "Flooring", "Roof", "Waterfront", "Sewer",
    "Pool", "Garden", "Balcony",
]
EXPECTED_TOTALS = {
    "original_rows": 44_896,
    "duplicate_copies": 4_082,
    "approved_removals": 4_360,
    "final_rows": 40_536,
    "review_only_rows": 1_714,
}


def validate_columns(df: pd.DataFrame) -> None:
    """Require the original feature schema before evaluating audit rules."""
    missing = [column for column in ORIGINAL_COLUMNS if column not in df.columns]
    extra = [column for column in df.columns if column not in ORIGINAL_COLUMNS]
    if missing or extra or not df.columns.is_unique:
        raise ValueError(f"Unexpected housing schema. Missing: {missing}; extra: {extra}; "
                         f"duplicate column names: {not df.columns.is_unique}")


def load_data(csv_path: Path = RAW_PATH) -> pd.DataFrame:
    """Read raw data with the audit's encoding and validate required columns."""
    df = pd.read_csv(csv_path, encoding="latin1")
    validate_columns(df)
    return df


def apply_approved_removals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the audit masks and remove each approved source row only once."""
    validate_columns(df)
    approved = flag_approved_removals(df)
    # Retain source order and every original column; no feature values are changed.
    cleaned = df.loc[~approved.any(axis=1)].copy().reset_index(drop=True)
    return cleaned, approved


def validate_cleaning(original: pd.DataFrame, cleaned: pd.DataFrame,
                      approved: pd.DataFrame) -> None:
    """Reject altered values, incorrect exclusions, or remaining approved rows."""
    validate_columns(original)
    validate_columns(cleaned)
    expected_flags = flag_approved_removals(original)
    try:
        pd.testing.assert_frame_equal(approved, expected_flags)
        expected = original.loc[~approved.any(axis=1)].reset_index(drop=True)
        pd.testing.assert_frame_equal(cleaned, expected)
    except AssertionError as error:
        raise ValueError("Cleaning must remove exactly the approved rows and preserve "
                         "all retained values, column order, and reset index.") from error
    if cleaned.duplicated().any():
        raise ValueError("Exact duplicate rows remain after approved removals.")
    if flag_approved_removals(cleaned).any(axis=1).any():
        raise ValueError("Approved-removal records remain in the cleaned data.")


def validate_audit_totals(original: pd.DataFrame, cleaned: pd.DataFrame,
                          approved: pd.DataFrame) -> dict[str, int]:
    """Check real-data findings against the approved audit, never force counts."""
    review = flag_review_candidates(original, approved)
    totals = {
        "original_rows": len(original),
        "duplicate_copies": int(original.duplicated().sum()),
        "approved_removals": int(approved.any(axis=1).sum()),
        "final_rows": len(cleaned),
        "review_only_rows": int(review.any(axis=1).sum()),
    }
    for name, expected in EXPECTED_TOTALS.items():
        if totals[name] != expected:
            raise ValueError(f"Audit mismatch for {name}: expected {expected:,}, "
                             f"observed {totals[name]:,}. No output saved.")
    for rule, expected in EXPECTED_COUNTS.items():
        actual = int(approved[rule].sum())
        if actual != expected:
            raise ValueError(f"Audit mismatch for {rule}: expected {expected}, "
                             f"observed {actual}. No output saved.")
    return totals


def save_cleaned_data(cleaned: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save a separate UTF-8 CSV without an index; protect the raw input path."""
    output_path = Path(output_path)
    if output_path.resolve() == RAW_PATH.resolve():
        raise ValueError("The cleaned output must not overwrite the raw dataset.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8")


def write_summary(totals: dict[str, int], approved: pd.DataFrame,
                  summary_path: Path = SUMMARY_PATH) -> None:
    """Write only execution counts and validation results, not another audit."""
    lines = [
        "Canada housing cleaning summary",
        f"Original rows: {totals['original_rows']:,} (23 columns)",
        f"Exact duplicate copies: {totals['duplicate_copies']:,}",
        f"Unique approved removals: {totals['approved_removals']:,}",
        f"Final rows: {totals['final_rows']:,}",
        f"Review-only records retained: {totals['review_only_rows']:,}",
        "Rule counts overlap; removed rows are counted once by source row.",
        "\nApproved removal counts:", approved.sum().to_string(),
        "\nPASS: expected audit totals and rule counts match.",
        "PASS: no approved-removal records or exact duplicates remain.",
        "PASS: all original columns and retained values preserved; index reset.",
        "PASS: review-only records retained; missing values left unchanged.",
        "No imputation, scaling, encoding, feature engineering, or training performed.",
        "City capitalization unchanged; canonical spellings remain to be specified.",
        "Raw dataset and audit files were not written to.",
        f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} (UTF-8; index=False)",
    ]
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Validate all cleaning decisions before writing the dataset and summary."""
    original = load_data()
    cleaned, approved = apply_approved_removals(original)
    validate_cleaning(original, cleaned, approved)
    totals = validate_audit_totals(original, cleaned, approved)
    save_cleaned_data(cleaned)
    write_summary(totals, approved)
    print(f"Removed {totals['approved_removals']:,} approved rows; "
          f"saved {len(cleaned):,} rows to {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
