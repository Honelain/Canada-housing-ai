"""Check partition identity, reproducibility, and lossless CSV exports."""

from pathlib import Path
import ast
import json
import tempfile
import unittest

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.split_dataset import SOURCE_PATH, load_data, save_splits, split_data, validate_split


class SplitDatasetTests(unittest.TestCase):
    def setUp(self):
        # Repeated values and index labels must not confuse source row identity.
        self.source = pd.DataFrame({
            "City": ["Montréal", "Example"] * 5,
            "Price": [123456.78901234567, 500000.0] * 5,
            "Basement": [None, "Finished"] * 5,
        }, index=[0] * 10)

    def test_reproducible_exact_partition_and_no_mutation(self):
        before = self.source.copy(deep=True)
        development, test = split_data(self.source)
        expected_dev, expected_test = train_test_split(
            self.source, test_size=0.2, random_state=42
        )
        self.assertEqual((len(development), len(test)), (8, 2))
        pd.testing.assert_frame_equal(development, expected_dev)
        pd.testing.assert_frame_equal(test, expected_test)
        again_dev, again_test = split_data(self.source)
        pd.testing.assert_frame_equal(development, again_dev)
        pd.testing.assert_frame_equal(test, again_test)
        pd.testing.assert_frame_equal(self.source, before)

    def test_unique_source_positions_cover_both_partitions(self):
        source = self.source.reset_index(drop=True)
        development, test = split_data(source)
        self.assertFalse(set(development.index) & set(test.index))
        self.assertEqual(sorted([*development.index, *test.index]), list(range(len(source))))

    def test_validation_rejects_changed_values_and_duplicated_partition(self):
        development, test = split_data(self.source)
        changed = development.copy()
        changed.iloc[0, 1] = 1
        with self.assertRaises(ValueError):
            validate_split(self.source, changed, test)
        with self.assertRaises(ValueError):
            validate_split(self.source, development, development.iloc[:2])

    def test_csv_round_trip_and_path_protection(self):
        development, test = split_data(self.source)
        with tempfile.TemporaryDirectory() as directory:
            dev_path = Path(directory) / "development.csv"
            test_path = Path(directory) / "test.csv"
            save_splits(self.source, development, test, dev_path, test_path)
            pd.testing.assert_frame_equal(load_data(dev_path), development.reset_index(drop=True))
            pd.testing.assert_frame_equal(load_data(test_path), test.reset_index(drop=True))
            with self.assertRaises(ValueError):
                save_splits(self.source, development, test, dev_path, dev_path)
            with self.assertRaises(ValueError):
                save_splits(self.source, development, test, SOURCE_PATH, test_path)

    def test_small_input_fails_clearly_and_rounding_is_expected(self):
        with self.assertRaises(ValueError):
            split_data(self.source.iloc[:1])
        development, test = split_data(self.source.iloc[:7])
        self.assertEqual((len(development), len(test)), (5, 2))

    def test_notebook_reads_only_development_partition(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/housing_eda.ipynb"
        notebook = json.loads(notebook_path.read_text())
        reads = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "read_csv":
                        reads.append(ast.unparse(node.args[0]))
        self.assertEqual(reads, ["DEVELOPMENT_PATH"])


if __name__ == "__main__":
    unittest.main()
