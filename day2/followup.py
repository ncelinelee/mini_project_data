"""개발 35셀에서만 상대 용량 피처와 nested group CV를 비교한다.

기존 평가 결과와 선정 모델은 덮어쓰지 않는다. Batch2/3 및 hold-out은
추가 실험의 피처/파라미터 선택과 점수 계산에 사용하지 않는다.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .experiment import ROOT, load_design
from .modeling import build_estimator, make_splits, metrics

FEATURES = {
    "original": ["deltaq_log10var", "qd_median", "qd_change100_10", "temp_mean"],
    "relative": ["relative_deltaq_log10var", "relative_qd_change", "temp_mean"],
}
GRID = [{"alpha": alpha, "l1_ratio": ratio}
        for alpha in (0.001, 0.01, 0.1) for ratio in (0.2, 0.5, 0.8)]


def relative_features(rows):
    """초기 100사이클 용량 중앙값으로 정규화. 전체 수명 정보는 쓰지 않는다."""
    result = rows.copy()
    capacity = result.qd_median.to_numpy(dtype=float)
    if not np.all(np.isfinite(capacity) & (capacity > 0)):
        raise ValueError("상대 피처에는 양의 유한한 초기 용량이 필요합니다.")
    result["relative_qd_change"] = result.qd_change100_10 / capacity
    # log10 Var(ΔQ / Q_early) = log10 Var(ΔQ) - 2 log10 Q_early.
    result["relative_deltaq_log10var"] = result.deltaq_log10var - 2 * np.log10(capacity)
    return result


def grouped_splits(rows, n_splits):
    folds = list(GroupKFold(n_splits=n_splits).split(rows, groups=rows.policy))
    for train, valid in folds:
        if not set(rows.iloc[train].policy).isdisjoint(rows.iloc[valid].policy):
            raise ValueError("충전 정책이 학습과 검증에 중복됐습니다.")
    return folds


def choose_parameters(rows, features):
    """외부 폴드의 학습 행만 받아 내부 3-fold에서 파라미터를 정한다."""
    folds = grouped_splits(rows, 3)
    scored = []
    for params in GRID:
        scores = []
        for train, valid in folds:
            estimator = build_estimator("ElasticNet", "ln", params)
            estimator.fit(rows.iloc[train][features], rows.iloc[train].cycle_life)
            pred = estimator.predict(rows.iloc[valid][features])
            scores.append(metrics(rows.iloc[valid].cycle_life, pred)["mape_pct"])
        scored.append((float(np.mean(scores)), params))
    return min(scored, key=lambda item: item[0])


def compare_development(df):
    _, holdout = load_design()
    # 분리 후부터는 개발 35셀만 실험 함수로 전달한다.
    dev, _, _, _ = make_splits(df.loc[df.batch == "Batch1"].copy(), holdout)
    dev = relative_features(dev)
    outer = grouped_splits(dev, 5)
    records, predictions, audit = [], [], []
    for fold, (train, valid) in enumerate(outer, 1):
        training, validation = dev.iloc[train], dev.iloc[valid]
        specs = [
            ("mean_baseline", "Mean", "identity", ["deltaq_log10var"], {}, None),
            ("reference_fixed", "ElasticNet", "ln", FEATURES["original"],
             {"alpha": 0.01, "l1_ratio": 0.5}, None),
        ]
        for definition, features in FEATURES.items():
            inner_score, params = choose_parameters(training, features)
            specs.append((f"nested_{definition}", "ElasticNet", "ln", features, params, inner_score))
        for method, model, transform, features, params, inner_score in specs:
            estimator = build_estimator(model, transform, params)
            estimator.fit(training[features], training.cycle_life)
            pred = estimator.predict(validation[features])
            records.append({"method": method, "fold": fold,
                            "n_train": len(training), "n_valid": len(validation),
                            **metrics(validation.cycle_life, pred),
                            "inner_mean_mape_pct": inner_score,
                            "parameters": json.dumps(params, sort_keys=True)})
            predictions.extend({"method": method, "fold": fold, "cell_id": cell,
                                "actual": float(actual), "prediction": float(value)}
                               for cell, actual, value in zip(validation.cell_id, validation.cycle_life, pred))
        inner = grouped_splits(training, 3)
        audit.append({"fold": fold, "train_ids": training.cell_id.tolist(),
                      "validation_ids": validation.cell_id.tolist(),
                      "train_policies": sorted(training.policy.unique()),
                      "validation_policies": sorted(validation.policy.unique()),
                      "inner_folds": [{"train_ids": training.iloc[a].cell_id.tolist(),
                                       "validation_ids": training.iloc[b].cell_id.tolist()}
                                      for a, b in inner]})
    fold_scores = pd.DataFrame(records)
    summary = (fold_scores.groupby("method", sort=False)
               .agg(mean_mape_pct=("mape_pct", "mean"), sd_mape_pct=("mape_pct", "std"),
                    mean_mae_cycles=("mae_cycles", "mean"), mean_rmse_cycles=("rmse_cycles", "mean"))
               .reset_index())
    # 동일 개발 표본·고정 모델로 분리 방식의 영향만 관찰한다. 선택에는 사용하지 않는다.
    split_records = []
    for strategy, folds in [("policy_group", outer),
                            ("random_cell", list(KFold(5, shuffle=True, random_state=42).split(dev)))]:
        for fold, (train, valid) in enumerate(folds, 1):
            estimator = build_estimator("ElasticNet", "ln", {"alpha": 0.01, "l1_ratio": 0.5})
            estimator.fit(dev.iloc[train][FEATURES["original"]], dev.iloc[train].cycle_life)
            pred = estimator.predict(dev.iloc[valid][FEATURES["original"]])
            overlap = set(dev.iloc[train].policy) & set(dev.iloc[valid].policy)
            split_records.append({"strategy": strategy, "fold": fold,
                                  "overlapping_policy_count": len(overlap),
                                  **metrics(dev.iloc[valid].cycle_life, pred)})
    return summary, fold_scores, pd.DataFrame(predictions), pd.DataFrame(split_records), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "day2/results/features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "day2/followup_results")
    args = parser.parse_args()
    df = pd.read_csv(args.features)
    summary, folds, predictions, strategies, audit = compare_development(df)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("summary", summary), ("fold_scores", folds),
                        ("predictions", predictions), ("split_strategies", strategies)]:
        frame.to_csv(args.output / f"{name}.csv", index=False)
    manifest = {
        "data_scope": "Batch1 development 35 cells only",
        "evaluation": "outer 5-fold GroupKFold; tuning inside 3-fold GroupKFold",
        "selection_metric": "mean fold MAPE (%)", "grid": GRID,
        "reference_limit": "Fixed reference was selected on these development cells in the earlier experiment; its score is not independent.",
        "independence_limit": "Follow-up motivated by previously observed Batch2 errors; no new independent test claim.",
        "external_scoring_performed": False,
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "supporting_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [ROOT / "day2/modeling.py", ROOT / "day2/experiment.py",
                         ROOT / "day2/config/holdout.json"]
        },
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        "folds": audit,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
