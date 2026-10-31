"""Small examples exercising the shared audit decisions and cleaning safeguards."""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src.cleaning.cleaning import (
    ORIGINAL_COLUMNS, RAW_PATH, apply_approved_removals, save_cleaned_data,
    validate_audit_totals, validate_cleaning, validate_columns,
)


def properties():
    base = dict.fromkeys(ORIGINAL_COLUMNS, None)
    base.update({"City": "Example", "Province": "ON", "Property Type": "Single Family",
                 "Price": 500000, "Bedrooms": 3, "Bathrooms": 2,
                 "Square Footage": 1500, "Acreage": 1})
    return pd.DataFrame([
        base, base.copy(),
        {**base, "City": "Small", "Square Footage": 450, "Bathrooms": 0},
        {**base, "City": "London", "Property Type": "MultiFamily", "Price": 3450888,
         "Bedrooms": 12, "Bathrooms": 26, "Square Footage": 16000},
        {**base, "City": "Rural", "Price": 6000000, "Acreage": 600},
    ], columns=ORIGINAL_COLUMNS)


class CleaningTests(unittest.TestCase):
    def test_overlap_duplicates_and_review_retention(self):
        df = properties()
        before = df.copy(deep=True)
        cleaned, flags = apply_approved_removals(df)
        self.assertEqual(int(flags.any(axis=1).sum()), 2)
        self.assertTrue(flags.loc[2, "Single Family with bedrooms but zero bathrooms"])
        self.assertTrue(flags.loc[2, "Bedrooms >= 2 and 400 <= area < 500"])
        self.assertEqual(cleaned.City.tolist(), ["Example", "London", "Rural"])
        self.assertEqual(cleaned.index.tolist(), [0, 1, 2])
        pd.testing.assert_frame_equal(df, before)
        validate_cleaning(df, cleaned, flags)

    def test_specific_property_does_not_remove_whole_city(self):
        df = properties().iloc[[0]].copy()
        df.loc[0, "City"] = "Langley"
        df.loc[0, "Province"] = "BC"
        cleaned, flags = apply_approved_removals(df)
        self.assertEqual(len(cleaned), 1)
        self.assertFalse(flags.any(axis=1).any())

    def test_validation_rejects_changed_values_and_wrong_rows(self):
        df = properties()
        cleaned, flags = apply_approved_removals(df)
        changed = cleaned.copy()
        changed.loc[0, "Price"] = 1
        with self.assertRaisesRegex(ValueError, "preserve"):
            validate_cleaning(df, changed, flags)
        with self.assertRaises(ValueError):
            validate_cleaning(df, df.reset_index(drop=True), flags)
        with self.assertRaisesRegex(ValueError, "Audit mismatch"):
            validate_audit_totals(df, cleaned, flags)

    def test_schema_and_raw_path_are_protected(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_columns(properties().drop(columns="Bathrooms"))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            save_cleaned_data(properties(), RAW_PATH)

    def test_save_has_original_columns_and_missing_values(self):
        cleaned, _ = apply_approved_removals(properties())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "processed" / "cleaned.csv"
            save_cleaned_data(cleaned, output)
            loaded = pd.read_csv(output)
            self.assertEqual(loaded.columns.tolist(), ORIGINAL_COLUMNS)
            self.assertEqual(loaded.Basement.isna().sum(), len(cleaned))


if __name__ == "__main__":
    unittest.main()
