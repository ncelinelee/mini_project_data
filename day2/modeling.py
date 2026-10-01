"""동일한 정책 분리에서 피처군·목표 변환·Grid 조합을 비교한다."""
import json
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def metrics(y, pred):
    if not np.all(np.isfinite(pred)):
        raise ValueError("예측에 NaN 또는 무한대가 있습니다.")
    return {"mape_pct": float(mean_absolute_percentage_error(y, pred) * 100),
            "mae_cycles": float(mean_absolute_error(y, pred)),
            "rmse_cycles": float(np.sqrt(mean_squared_error(y, pred)))}


def make_splits(df, holdout):
    batch1 = df.loc[(df.batch == "Batch1") & df.target_available].reset_index(drop=True)
    dev_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                            .split(batch1, groups=batch1.policy))
    dev = batch1.iloc[dev_idx].reset_index(drop=True)
    valid = batch1.iloc[val_idx].reset_index(drop=True)
    if (dev.cell_id.tolist() != holdout["development_cell_ids"] or
            valid.cell_id.tolist() != holdout["holdout_cell_ids"]):
        raise ValueError("DAY1에서 고정한 셀 분할과 다릅니다.")
    assert set(dev.policy).isdisjoint(valid.policy)
    folds = list(GroupKFold(n_splits=5, shuffle=False).split(dev, groups=dev.policy))
    fold_records = []
    for n, (train, test) in enumerate(folds, 1):
        assert set(dev.iloc[train].policy).isdisjoint(dev.iloc[test].policy)
        fold_records.append({"fold": n, "train_cell_ids": dev.iloc[train].cell_id.tolist(),
                             "validation_cell_ids": dev.iloc[test].cell_id.tolist(),
                             "train_policies": sorted(dev.iloc[train].policy.unique()),
                             "validation_policies": sorted(dev.iloc[test].policy.unique())})
    return dev, valid, folds, fold_records


def build_estimator(model, transform, params):
    if model == "Mean":
        estimator = DummyRegressor(strategy="mean")
    elif model == "Univariate":
        estimator = LinearRegression(fit_intercept=True)
    elif model == "Ridge":
        estimator = Ridge(fit_intercept=True, **params)
    elif model == "ElasticNet":
        estimator = ElasticNet(fit_intercept=True, max_iter=10000, tol=1e-4,
                               selection="cyclic", **params)
    elif model == "RandomForest":
        estimator = RandomForestRegressor(n_estimators=300, bootstrap=True,
                    max_features=1.0, criterion="squared_error", random_state=42,
                    n_jobs=1, **params)
    elif model == "GradientBoosting":
        estimator = GradientBoostingRegressor(min_samples_leaf=3, loss="squared_error",
                    subsample=1.0, n_iter_no_change=None, random_state=42, **params)
    else:
        raise ValueError(model)
    steps = []
    if model in ("Univariate", "Ridge", "ElasticNet"):
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    if transform == "ln":
        return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
    return pipeline


def fitted_pipeline(estimator):
    return estimator.regressor_ if isinstance(estimator, TransformedTargetRegressor) else estimator


def candidate_specs(design):
    candidates = []
    for transform in ("identity", "ln"):
        candidates.extend([
            {"model": "Mean", "feature_group": "baseline", "target_transform": transform,
             "features": ["deltaq_log10var"], "params": {}},
            {"model": "Univariate", "feature_group": "univariate", "target_transform": transform,
             "features": ["deltaq_log10var"], "params": {}},
        ])
        for group, features in design["feature_groups"].items():
            grids = design["tuning_grids"]
            for model, grid in [
                ("Ridge", {"alpha": grids["Ridge"]["alpha"]}),
                ("ElasticNet", {"alpha": grids["ElasticNet"][f"alpha_{transform}"],
                                "l1_ratio": grids["ElasticNet"]["l1_ratio"]}),
                ("RandomForest", {k: grids["RandomForest"][k] for k in ("max_depth", "min_samples_leaf")}),
                ("GradientBoosting", {k: grids["GradientBoosting"][k] for k in ("learning_rate", "n_estimators", "max_depth")}),
            ]:
                for params in ParameterGrid(grid):
                    candidates.append({"model": model, "feature_group": group,
                        "target_transform": transform, "features": features, "params": params})
    for i, spec in enumerate(candidates):
        spec["candidate_id"] = f"candidate-{i:03d}"
    if len(candidates) != 294:
        raise ValueError("290개 튜닝 조합과 4개 기준 후보가 필요합니다.")
    return candidates


def evaluate_candidate(spec, dev, folds):
    x, y = dev[spec["features"]], dev.cycle_life.to_numpy()
    oof = np.full(len(dev), np.nan)
    fold_results, warning_log = [], []
    with threadpool_limits(limits=1):
        for fold, (train, test) in enumerate(folds, 1):
            estimator = build_estimator(spec["model"], spec["target_transform"], spec["params"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(x.iloc[train], y[train])
            for warning in caught:
                warning_log.append({"candidate_id": spec["candidate_id"], "fold": fold,
                    "category": warning.category.__name__, "message": str(warning.message)})
            pred = estimator.predict(x.iloc[test])
            oof[test] = pred
            result = metrics(y[test], pred)
            iteration = getattr(fitted_pipeline(estimator).named_steps["model"], "n_iter_", None)
            result.update(fold=fold, n_train=len(train), n_validation=len(test),
                          n_iter=int(iteration) if iteration is not None else None)
            fold_results.append(result)
    result = {**spec, "params_json": json.dumps(spec["params"], sort_keys=True),
              "mean_cv_mape_pct": float(np.mean([r["mape_pct"] for r in fold_results])),
              "std_cv_mape_pct": float(np.std([r["mape_pct"] for r in fold_results], ddof=1)),
              "mean_cv_mae_cycles": float(np.mean([r["mae_cycles"] for r in fold_results])),
              "mean_cv_rmse_cycles": float(np.mean([r["rmse_cycles"] for r in fold_results])),
              "pooled_oof_mape_pct": metrics(y, oof)["mape_pct"],
              "warning_count": len(warning_log), "fold_results": fold_results}
    return result, oof, warning_log
