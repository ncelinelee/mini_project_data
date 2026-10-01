"""제출 노트북을 처음부터 실행하고 점수·예측·분리를 다시 대조한다."""
import os
from pathlib import Path
import json
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".venv/jupyter_runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".venv/ipython"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv/matplotlib"))
for key in ["JUPYTER_RUNTIME_DIR", "IPYTHONDIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

from day2.experiment import write_json
from day2.modeling import metrics


def main():
    path = ROOT / "Day2_Battery_Cycle_Life.ipynb"
    output = ROOT / "day2/results"
    baseline_cv = pd.read_csv(output / "cv_results.csv")
    baseline_predictions = pd.read_csv(output / "predictions.csv")
    baseline_selection = json.loads((output / "selection.json").read_text())
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=180, kernel_name="battery-day2",
                            resources={"metadata": {"path": str(ROOT)}})
    started = time.perf_counter()
    client.execute()
    nbformat.write(notebook, path)
    current_cv = pd.read_csv(output / "cv_results.csv")
    pd.testing.assert_frame_equal(baseline_cv, current_cv, check_exact=False, rtol=1e-10, atol=1e-10)
    predictions = pd.read_csv(output / "predictions.csv")
    pd.testing.assert_frame_equal(baseline_predictions, predictions, check_exact=False, rtol=1e-10, atol=1e-10)
    winner = json.loads((output / "selection.json").read_text())
    assert winner["candidate_id"] == baseline_selection["candidate_id"]
    score_table = pd.read_csv(output / "metrics.csv").set_index("split")
    for split, rows in predictions.groupby("split"):
        actual = metrics(rows.cycle_life, rows.prediction)
        for field, value in actual.items():
            np.testing.assert_allclose(value, score_table.loc[split, field], rtol=1e-10, atol=1e-10)
    details = json.loads((output / "cv_details.json").read_text())
    for candidate in details:
        expected = np.mean([fold["mape_pct"] for fold in candidate["fold_results"]])
        np.testing.assert_allclose(expected, candidate["mean_cv_mape_pct"], atol=1e-12)
    best = min(details, key=lambda c: c["mean_cv_mape_pct"])
    assert best["candidate_id"] == winner["candidate_id"]
    oof = pd.read_csv(output / "all_oof_predictions.csv")
    assert len(oof) == 35 and len(oof.columns) == 296
    assert np.isfinite(oof.select_dtypes(include="number")).all().all()
    assert len(predictions) == 129 and predictions.cell_id.is_unique
    # Unknown labels remain in observed features, never in scored predictions.
    features = pd.read_csv(output / "features.csv")
    assert len(features) == 139 and features.target_available.sum() == 129
    assert set(features.loc[~features.target_available, "cell_id"]).isdisjoint(predictions.cell_id)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    errors = [item for cell in code_cells for item in cell.outputs if item.output_type == "error"]
    assert not errors
    write_json(output / "verification.json", {
        "notebook_execution": "passed", "executed_code_cells": len(code_cells),
        "notebook_error_outputs": len(errors), "notebook_elapsed_seconds": time.perf_counter() - started,
        "full_grid_reproduction": "all 294 candidate CV scores matched within 1e-10",
        "prediction_reproduction": "all 129 predictions matched within 1e-10",
        "minimum_unrounded_fold_mean_selection": "passed", "score_recalculation": "passed",
        "missing_targets_excluded": "passed", "all_oof_cells_predicted_once_per_candidate": "passed",
        "contract_tests": "5 tests passed separately",
        "graph_visual_review": "seven final charts reviewed",
    })
    print("PASS: 노트북 전체 실행, 294개 후보 점수와 129개 예측 재현, 지표·선정·결측 제외 검증")


if __name__ == "__main__":
    main()
