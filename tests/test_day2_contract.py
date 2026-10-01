"""데이터 누출, 원래 사이클 번호, 점수 단위를 검증하는 계약 테스트."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from day2.data import SUMMARY_FIELDS, initial_features
from day2.experiment import load_design
from day2.modeling import build_estimator, candidate_specs, make_splits, metrics

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def fixture(self):
        cycle = np.arange(1, 151)
        measurements = {key: np.ones(150) for key in SUMMARY_FIELDS}
        for value in measurements.values():
            value[0] = 0
        grid = np.linspace(3.6, 2.0, 1000)
        q10 = np.linspace(0, 1, 1000)
        q100 = q10 + np.linspace(-.01, -.001, 1000)
        return cycle, measurements, q10, q100, grid

    def test_future_measurements_do_not_change_inputs(self):
        cycle, values, q10, q100, grid = self.fixture()
        a = initial_features(cycle, values, q10, q100, grid, "4C(80%)-4C", blank_first=True)
        for v in values.values():
            v[100:] = 1e9
        b = initial_features(cycle, values, q10, q100, grid, "4C(80%)-4C", blank_first=True)
        self.assertEqual(a, b)
        self.assertEqual(a["valid_QD_early100"], 99)
        self.assertEqual(values["QDischarge"][0], 0)  # raw preserved

    def test_original_cycle100_not_100th_valid_measurement(self):
        cycle, values, q10, q100, grid = self.fixture()
        values["QDischarge"][99] = 1.1
        values["QDischarge"][100] = 9
        a = initial_features(cycle, values, q10, q100, grid, "4C(80%)-4C", blank_first=True)
        self.assertAlmostEqual(a["qd_change100_10"], .1)

    def test_log_baseline_returns_cycles_and_geometric_mean(self):
        estimator = build_estimator("Mean", "ln", {})
        x = pd.DataFrame({"deltaq_log10var": [-5, -4]})
        estimator.fit(x, [100., 400.])
        pred = estimator.predict(x)
        np.testing.assert_allclose(pred, [200., 200.])
        self.assertAlmostEqual(metrics([100., 400.], pred)["mape_pct"], 75.)

    def test_scaler_fitted_only_on_fold_training_values(self):
        x = pd.DataFrame({"a": [1., 2., 1000.]})
        estimator = build_estimator("Ridge", "identity", {"alpha": 1})
        estimator.fit(x.iloc[:2], [100., 200.])
        self.assertIsInstance(estimator.named_steps["scale"], StandardScaler)
        self.assertEqual(estimator.named_steps["scale"].mean_[0], 1.5)
        estimator.predict(x.iloc[2:])
        self.assertEqual(estimator.named_steps["scale"].mean_[0], 1.5)

    def test_fixed_split_and_no_policy_overlap(self):
        design, holdout = load_design()
        reference = json.loads((ROOT / "day1/step03/candidate_features.json").read_text())
        dev, valid, folds, _ = make_splits(pd.DataFrame(reference), holdout)
        self.assertEqual((len(dev), len(valid)), (35, 11))
        all_validation = []
        for train, test in folds:
            self.assertTrue(set(dev.iloc[train].policy).isdisjoint(dev.iloc[test].policy))
            all_validation.extend(test.tolist())
        self.assertEqual(sorted(all_validation), list(range(35)))
        self.assertEqual(len(candidate_specs(design)) * len(folds), 1470)


if __name__ == "__main__":
    unittest.main()
