"""원본 추출·후보 탐색·최종 평가를 나누어 결과를 저장한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import BATCHES, INPUT_FIELDS, extract_features
from .modeling import (build_estimator, candidate_specs, evaluate_candidate,
                       fitted_pipeline, make_splits, metrics)

from .reporting import performance_tables

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "day2/results"


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False, default=json_default) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_design(root=ROOT):
    config_dir = Path(root) / "day2/config"
    return (json.loads((config_dir / "design.json").read_text()),
            json.loads((config_dir / "holdout.json").read_text()))


def prepare(data_dir=ROOT / "data-30", output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df, audit = extract_features(data_dir)
    df.to_csv(output / "features.csv", index=False)
    write_json(output / "processing_audit.json", audit)
    # The old table is an independent DAY1 parity check, never the extraction source.
    reference = ROOT / "day1/step03/candidate_features.json"
    if reference.exists():
        old = pd.DataFrame(json.loads(reference.read_text())).set_index("cell_id")
        new = df.set_index("cell_id")
        for field in INPUT_FIELDS + ["cycle_life"]:
            np.testing.assert_allclose(new[field], old.loc[new.index, field].astype(float),
                                       rtol=1e-12, atol=1e-12, equal_nan=True)
        print("DAY1 피처와 원본 재생성 피처가 일치합니다.", flush=True)
    write_json(output / "data_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"batch": batch, "filename": name,
                   "bytes": (Path(data_dir) / name).stat().st_size} for batch, name in BATCHES.items()],
        "feature_sha256": digest(output / "features.csv"),
        "total_cells": len(df), "labelled_cells": int(df.target_available.sum()),
        "day1_parity_check": "passed" if reference.exists() else "not_available",
        "feature_window": "original cycles 1-100", "interpolation": "none; common voltage grid",
    })
    return df


def tune(df=None, output=DEFAULT_OUTPUT, jobs=4):
    output = Path(output)
    if df is None:
        df = pd.read_csv(output / "features.csv")
    design, holdout = load_design()
    dev, _, folds, fold_records = make_splits(df, holdout)
    write_json(output / "cv_splits.json", fold_records)
    specs = candidate_specs(design)
    print(f"Grid Search 시작: {len(specs)}개 후보 × 5-fold = 1,470회 학습", flush=True)
    started = time.perf_counter()
    # Only the 35 development rows are passed to workers. Held-out targets cannot score candidates.
    outputs = Parallel(n_jobs=jobs, verbose=5, backend="loky")(
        delayed(evaluate_candidate)(spec, dev, folds) for spec in specs)
    results = [item[0] for item in outputs]
    table = pd.DataFrame([{k: v for k, v in item.items() if k not in
                           ("fold_results", "features", "params")} for item in results])
    table = table.sort_values("mean_cv_mape_pct", kind="stable").reset_index(drop=True)
    table.to_csv(output / "cv_results.csv", index=False)
    write_json(output / "cv_details.json", results)
    write_json(output / "fit_warnings.json", [w for _, _, log in outputs for w in log])
    oof = pd.DataFrame({"cell_id": dev.cell_id.to_numpy(), "cycle_life": dev.cycle_life.to_numpy(),
                        **{result["candidate_id"]: pred for result, pred, _ in outputs}})
    oof.to_csv(output / "all_oof_predictions.csv", index=False)
    best_id = table.iloc[0].candidate_id
    winner = next(item for item in results if item["candidate_id"] == best_id)
    winner.update(selection_rule=design["final_selection_rule"],
                  score_aggregation=design["cv_score_aggregation"],
                  tuning_fit_count=1450, baseline_fit_count=20,
                  total_cv_fits=1470, elapsed_seconds=time.perf_counter() - started)
    # Freeze the winner before opening evaluation splits.
    write_json(output / "selection.json", winner)
    print(f"CV 선정 완료: {winner['model']} / {winner['feature_group']} / "
          f"{winner['target_transform']} / MAPE {winner['mean_cv_mape_pct']:.3f}%", flush=True)
    return winner


def predict_rows(estimator, rows, features, split):
    pred = estimator.predict(rows[features])
    result = rows[["cell_id", "batch", "policy", "cycle_life"]].copy()
    result["split"] = split
    result["prediction"] = pred
    result["error_cycles"] = pred - result.cycle_life
    result["absolute_error_cycles"] = np.abs(result.error_cycles)
    result["ape_pct"] = result.absolute_error_cycles / result.cycle_life * 100
    return result


def evaluate(df=None, output=DEFAULT_OUTPUT):
    output = Path(output)
    if df is None:
        df = pd.read_csv(output / "features.csv")
    _, holdout = load_design()
    dev, valid, _, _ = make_splits(df, holdout)
    selection_path = output / "selection.json"
    frozen_hash = digest(selection_path)
    winner = json.loads(selection_path.read_text())
    estimator = build_estimator(winner["model"], winner["target_transform"], winner["params"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(dev[winner["features"]], dev.cycle_life)
    write_json(output / "final_fit_warnings.json", [
        {"category": w.category.__name__, "message": str(w.message)} for w in caught])
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"estimator": estimator, "features": winner["features"],
                 "selection": winner, "sklearn_version": importlib.metadata.version("scikit-learn")},
                model_dir / "selected_model.joblib")
    evaluated, score_rows = [], []
    splits = [("Fit_Batch1", dev), ("Valid", valid),
              ("Test_Batch2", df.loc[(df.batch == "Batch2") & df.target_available]),
              ("Additional_Batch3", df.loc[(df.batch == "Batch3") & df.target_available])]
    for name, rows in splits:
        predictions = predict_rows(estimator, rows, winner["features"], name)
        evaluated.append(predictions)
        score_rows.append({"split": name, "n": len(rows),
                           **metrics(predictions.cycle_life, predictions.prediction)})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(output / "metrics.csv", index=False)
    predictions = pd.concat(evaluated, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    oof = pd.read_csv(output / "all_oof_predictions.csv")
    chosen_oof = oof[["cell_id", "cycle_life", winner["candidate_id"]]].rename(
        columns={winner["candidate_id"]: "prediction"})
    chosen_oof.to_csv(output / "selected_oof_predictions.csv", index=False)
    intervals = [predictions.cycle_life < 500,
                 predictions.cycle_life.between(500, 1000), predictions.cycle_life > 1000]
    predictions["life_group"] = np.select(intervals, ["<500", "500-1000", ">1000"], default="unknown")
    groups = []
    for (split, group), sub in predictions.groupby(["split", "life_group"]):
        groups.append({"split": split, "life_group": group, "n": len(sub),
                       **metrics(sub.cycle_life, sub.prediction),
                       "bias_cycles": float(sub.error_cycles.mean())})
    pd.DataFrame(groups).to_csv(output / "subgroup_metrics.csv", index=False)
    predictions.nlargest(10, "ape_pct").to_csv(output / "largest_errors.csv", index=False)
    ids = ["Batch3-C07", "Batch3-C38", "Batch3-C45"]
    predictions.loc[predictions.cell_id.isin(ids)].to_csv(output / "long_life_cells.csv", index=False)
    shift_rows = []
    for name, rows in splits[1:]:
        for feature in winner["features"]:
            lo, hi = dev[feature].min(), dev[feature].max()
            shift_rows.append({"split": name, "feature": feature, "n": len(rows),
                "development_min": lo, "development_max": hi,
                "development_median": dev[feature].median(),
                "evaluation_median": rows[feature].median(),
                "below_development_min": int((rows[feature] < lo).sum()),
                "above_development_max": int((rows[feature] > hi).sum())})
    pd.DataFrame(shift_rows).to_csv(output / "feature_range_shift.csv", index=False)
    mandatory, additional, gaps = performance_tables(scores, winner)
    mandatory.to_csv(output / "model_performance.csv", index=False)
    additional.to_csv(output / "model_performance_batch3.csv", index=False)
    write_json(output / "performance_gaps.json", gaps)
    # Interpretation only: this does not change model selection after evaluation.
    importance = permutation_importance(estimator, valid[winner["features"]], valid.cycle_life,
        scoring="neg_mean_absolute_percentage_error", n_repeats=30, random_state=42, n_jobs=1)
    pd.DataFrame({"feature": winner["features"], "increase_mape_pp": importance.importances_mean * 100,
                  "repeat_sd_pp": importance.importances_std * 100}).to_csv(
                      output / "permutation_importance_valid.csv", index=False)
    pipeline = fitted_pipeline(estimator)
    # JSON-friendly parameter dump records every API default used by the installed version.
    params = {k: v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
              for k, v in estimator.get_params(deep=True).items()}
    write_json(output / "estimator_parameters.json", params)
    if hasattr(pipeline.named_steps["model"], "coef_"):
        pd.DataFrame({"feature": winner["features"],
                      "coefficient_standardized_x": pipeline.named_steps["model"].coef_}).to_csv(
                          output / "linear_coefficients.csv", index=False)
    # Confirm the serialized artifact gives exactly the same predictions.
    restored = joblib.load(model_dir / "selected_model.joblib")
    np.testing.assert_allclose(restored["estimator"].predict(valid[restored["features"]]),
                               evaluated[1].prediction, rtol=1e-12, atol=1e-12)
    assert digest(selection_path) == frozen_hash
    write_json(output / "run_manifest.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "h5py", "scikit-learn", "joblib", "matplotlib"]},
        "selection_sha256": frozen_hash, "feature_sha256": digest(output / "features.csv"),
        "design_sha256": digest(ROOT / "day2/config/design.json"),
        "holdout_sha256": digest(ROOT / "day2/config/holdout.json"),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "day2").glob("*.py"))},
        "selection_unchanged_during_evaluation": True,
        "model_reload_prediction_check": "passed", "negative_prediction_count": int((predictions.prediction < 0).sum()),
        "convergence_warning_count_cv": sum(w["category"] == "ConvergenceWarning" for w in
                                                json.loads((output / "fit_warnings.json").read_text())),
    })
    print(scores.to_string(index=False), flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description="DAY1 설계에 따른 총수명 회귀 실험")
    parser.add_argument("--stage", choices=["all", "prepare", "tune", "evaluate", "plots"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data-30")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage in ("all", "prepare"):
        prepare(args.data_dir, args.output)
    if args.stage in ("all", "tune"):
        tune(output=args.output, jobs=args.jobs)
    if args.stage in ("all", "evaluate"):
        evaluate(output=args.output)
    if args.stage in ("all", "plots"):
        from .plots import make_figures
        make_figures(args.output)


if __name__ == "__main__":
    main()
