"""Regression checks for the audit's remaining report outputs."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.audit import audit_pipeline as audit


class AuditReportingTests(unittest.TestCase):
    def test_run_preserves_results_and_never_uses_excel(self):
        original = audit.load_data()
        before = original.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(audit, "load_data", return_value=original), \
                 patch.object(pd, "ExcelWriter", side_effect=AssertionError("Excel export called")), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = audit.run_audit(directory)
            self.assertEqual(set(result), {"approved_removals", "review_candidates"})
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {"audit_summary.txt", "flagged_records.csv"})
            approved = result["approved_removals"]
            review = result["review_candidates"]
            self.assertEqual(original.shape, (44896, 23))
            self.assertEqual(int(original.duplicated().sum()), 4082)
            self.assertEqual(int(approved.any(axis=1).sum()), 4360)
            self.assertEqual(int(review.any(axis=1).sum()), 1714)
            self.assertEqual(int((~approved.any(axis=1)).sum()), 40536)
            for rule, expected in audit.EXPECTED_COUNTS.items():
                self.assertEqual(int(approved[rule].sum()), expected)
            summary = (Path(directory) / "audit_summary.txt").read_text()
            self.assertNotIn("MISMATCH", summary)
            records = pd.read_csv(Path(directory) / "flagged_records.csv")
            self.assertEqual(records.shape, (6074, 26))
            self.assertTrue(records.source_index.is_unique)
            self.assertEqual(records.audit_status.value_counts().to_dict(),
                             {"approved_removal": 4360, "review_only": 1714})
        pd.testing.assert_frame_equal(original, before)


if __name__ == "__main__":
    unittest.main()
