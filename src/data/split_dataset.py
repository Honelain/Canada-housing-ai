"""Create a reproducible listing-level holdout before development-only EDA."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "data/processed/cleaned_housing.csv"
DEVELOPMENT_PATH = PROJECT_ROOT / "data/processed/development_housing.csv"
TEST_PATH = PROJECT_ROOT / "data/processed/test_housing.csv"
RANDOM_STATE = 42


def load_data(csv_path: Path = SOURCE_PATH) -> pd.DataFrame:
    """Read the cleaner's UTF-8 output using the pinned pandas parser."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split source positions, preserving values even if index labels repeat."""
    development_positions, test_positions = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE
    )
    development = df.iloc[development_positions].copy()
    test = df.iloc[test_positions].copy()
    validate_split(df, development, test)
    return development, test


def validate_split(source: pd.DataFrame, development: pd.DataFrame,
                   test: pd.DataFrame) -> None:
    """Require the exact seeded partition, including row order and all values."""
    development_positions, test_positions = train_test_split(
        np.arange(len(source)), test_size=0.2, random_state=RANDOM_STATE
    )
    all_positions = np.concatenate([development_positions, test_positions])
    if (len(development) + len(test) != len(source)
            or not np.array_equal(np.sort(all_positions), np.arange(len(source)))):
        raise ValueError("Partitions must contain every source position exactly once.")
    try:
        pd.testing.assert_frame_equal(development, source.iloc[development_positions], check_exact=True)
        pd.testing.assert_frame_equal(test, source.iloc[test_positions], check_exact=True)
    except AssertionError as error:
        raise ValueError("Partitions must preserve the seeded source rows and values.") from error


def save_splits(source: pd.DataFrame, development: pd.DataFrame,
                test: pd.DataFrame, development_path: Path = DEVELOPMENT_PATH,
                test_path: Path = TEST_PATH) -> None:
    """Validate, save separate CSVs, and verify values survive serialization."""
    validate_split(source, development, test)
    paths = [Path(development_path), Path(test_path)]
    if len({SOURCE_PATH.resolve(), *(path.resolve() for path in paths)}) != 3:
        raise ValueError("Output paths must be distinct and cannot overwrite the source.")
    for partition, path in zip((development, test), paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        partition.to_csv(path, index=False, encoding="utf-8")
        # Indices are partition metadata, not predictor columns.
        restored = load_data(path)
        pd.testing.assert_frame_equal(restored, partition.reset_index(drop=True), check_exact=True)


def main() -> None:
    """Write the fixed split; reserve the test CSV for final evaluation only."""
    source = load_data()
    development, test = split_data(source)
    save_splits(source, development, test)
    print(f"Source: {len(source):,} rows; {len(source.columns)} columns")
    print(f"Development: {len(development):,} rows -> {DEVELOPMENT_PATH}")
    print(f"Untouched test: {len(test):,} rows -> {TEST_PATH}")
    print("PASS: seeded split, complete row coverage, and CSV values verified.")
    print("Use only development_housing.csv for EDA and model development.")


if __name__ == "__main__":
    main()
