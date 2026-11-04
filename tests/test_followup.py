"""후속 실험의 외부 데이터 격리와 피처 정의를 검증한다."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from day2.followup import compare_development, relative_features

ROOT = Path(__file__).resolve().parents[1]


class FollowupTests(unittest.TestCase):
    def test_relative_capacity_scale_invariance(self):
        rows = pd.DataFrame({"qd_median": [1., 2.], "qd_change100_10": [.1, .2],
                             "deltaq_log10var": [-4., -4. + 2 * np.log10(2)]})
        result = relative_features(rows)
        np.testing.assert_allclose(result.relative_qd_change, [.1, .1])
        np.testing.assert_allclose(result.relative_deltaq_log10var, [-4., -4.])

    def test_nonpositive_capacity_is_rejected(self):
        for value in (0., -1., np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_features(pd.DataFrame({"qd_median": [value]}))

    def test_evaluation_targets_cannot_change_development_results(self):
        from day2.experiment import load_design
        _, holdout = load_design()
        rows = pd.read_csv(ROOT / "day2/results/features.csv")
        altered = rows.copy()
        external = ~altered.cell_id.isin(holdout["development_cell_ids"])
        altered.loc[external, "cycle_life"] = 1e9
        original = compare_development(rows)
        changed = compare_development(altered)
        for before, after in zip(original[:4], changed[:4]):
            pd.testing.assert_frame_equal(before, after)
        self.assertEqual(original[4], changed[4])
        self.assertEqual(set(original[2].cell_id), set(holdout["development_cell_ids"]))
        for fold in original[4]:
            self.assertTrue(set(fold["train_policies"]).isdisjoint(fold["validation_policies"]))
            for inner in fold["inner_folds"]:
                self.assertTrue(set(inner["train_ids"] + inner["validation_ids"]).issubset(fold["train_ids"]))


if __name__ == "__main__":
    unittest.main()
