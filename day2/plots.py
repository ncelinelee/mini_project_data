"""한 그래프가 한 비교에 답하도록 결과를 시각화한다. 영문 축은 OS에 독립적이다."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".venv/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Fit_Batch1": "#777777", "Valid": "#454545", "Test_Batch2": "#A56B3F",
          "Additional_Batch3": "#69754C"}
NAMES = {"Fit_Batch1": "Fit · Batch1 (same training cells)", "Valid": "Valid · Batch1",
         "Test_Batch2": "Test · Batch2", "Additional_Batch3": "Additional · Batch3"}
FEATURE_LABELS = {"deltaq_log10var": "log10 variance of Delta Q", "deltaq_min": "Minimum Delta Q",
    "qd_median": "Median discharge capacity", "qd_iqr": "IQR of discharge capacity",
    "qd_change100_10": "Capacity change: cycle 100 - 10", "temp_mean": "Mean temperature",
    "charge_median": "Median charging time", "policy_c1": "Stage 1 C-rate",
    "policy_c2": "Stage 2 C-rate", "policy_switch_pct": "Switching SOC (%)"}


def save(fig, directory, name):
    fig.tight_layout(pad=1.6)
    fig.savefig(directory / name, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_figures(output):
    output = Path(output)
    dest = output / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold"})
    winner = json.loads((output / "selection.json").read_text())
    cv = pd.read_csv(output / "cv_results.csv")
    best = cv.sort_values("mean_cv_mape_pct").drop_duplicates("model").sort_values("mean_cv_mape_pct")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(np.arange(len(best)), best.mean_cv_mape_pct,
                  color=["#69754C" if cid == winner["candidate_id"] else "#777777" for cid in best.candidate_id], height=.55)
    ax.set_yticks(np.arange(len(best)), best.model)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in best.mean_cv_mape_pct], padding=6)
    ax.set_xlim(0, best.mean_cv_mape_pct.max() * 1.23)
    ax.set_xlabel("Mean of five policy-group CV MAPE scores (%) · lower is better")
    ax.set_title("Best CV result for each model", loc="left", pad=18)
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    save(fig, dest, "01_model_comparison.png")

    pred = pd.read_csv(output / "predictions.csv")
    scores = pd.read_csv(output / "metrics.csv").set_index("split")
    bounds = [250, max(2100, np.ceil(max(pred.cycle_life.max(), pred.prediction.max()) / 100) * 100)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for ax, split in zip(axes.ravel(), COLORS):
        sub = pred[pred.split == split]
        ax.scatter(sub.cycle_life, sub.prediction, color=COLORS[split], s=48, alpha=.8,
                   edgecolors="white", linewidth=.5)
        ax.plot(bounds, bounds, color="#999999", linestyle="--", linewidth=1)
        ax.set(xlim=bounds, ylim=bounds, xlabel="Actual life (cycles)", ylabel="Predicted life (cycles)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{NAMES[split]} · n={len(sub)}\nMAPE {scores.loc[split, 'mape_pct']:.2f}%", loc="left")
        ax.grid(alpha=.15)
    save(fig, dest, "02_actual_vs_predicted.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, split in zip(axes, list(COLORS)[1:]):
        sub = pred[pred.split == split]
        ax.scatter(sub.cycle_life, sub.ape_pct, color=COLORS[split], s=40, alpha=.8)
        for _, row in sub[sub.cell_id.isin(["Batch3-C07", "Batch3-C38", "Batch3-C45"])].iterrows():
            ax.annotate(row.cell_id.split("-")[1], (row.cycle_life, row.ape_pct),
                        xytext=(0, 9 if row.cell_id != "Batch3-C45" else -16),
                        textcoords="offset points", ha="center", fontsize=10)
        ax.set_title(f"{NAMES[split]} · n={len(sub)}", loc="left")
        ax.set_xlabel("Actual life (cycles)"); ax.grid(alpha=.15)
    axes[0].set_ylabel("Absolute percentage error (%)")
    axes[0].set_xlim(300, 2100)
    save(fig, dest, "03_error_by_lifetime.png")

    groups = pd.read_csv(output / "subgroup_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    group_names = ["<500", "500-1000", ">1000"]
    ymax = groups[groups.split != "Fit_Batch1"].mape_pct.max() * 1.3
    for ax, split in zip(axes, list(COLORS)[1:]):
        sub = groups[groups.split == split].set_index("life_group")
        values = [sub.loc[g, "mape_pct"] if g in sub.index else 0 for g in group_names]
        counts = [int(sub.loc[g, "n"]) if g in sub.index else 0 for g in group_names]
        bars = ax.bar(group_names, values, color=COLORS[split], width=.55)
        ax.bar_label(bars, labels=[f"{v:.1f}%\nn={n}" if n else "n=0" for v, n in zip(values, counts)], padding=5)
        ax.set_title(NAMES[split], loc="left"); ax.set_xlabel("Actual life group (cycles)")
        ax.set_ylim(0, ymax); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel("MAPE (%)")
    save(fig, dest, "04_lifetime_group_errors.png")

    selected = cv[(cv.model == winner["model"]) & (cv.feature_group == winner["feature_group"])
                  & (cv.target_transform == winner["target_transform"])].copy()
    selected["params"] = selected.params_json.map(json.loads)
    fig, ax = plt.subplots(figsize=(10, max(4, .4 * len(selected) + 1.8)))
    if winner["model"] in ("Ridge", "ElasticNet"):
        selected["alpha"] = selected.params.map(lambda p: p["alpha"])
        if winner["model"] == "ElasticNet":
            selected["l1_ratio"] = selected.params.map(lambda p: p["l1_ratio"])
            for (ratio, col) in zip(sorted(selected.l1_ratio.unique()), ["#454545", "#A56B3F", "#69754C"]):
                part = selected[selected.l1_ratio == ratio].sort_values("alpha")
                ax.plot(part.alpha, part.mean_cv_mape_pct, marker="o", color=col, label=f"l1_ratio={ratio}")
            ax.legend(frameon=False)
        else:
            part = selected.sort_values("alpha")
            ax.plot(part.alpha, part.mean_cv_mape_pct, marker="o", color="#454545")
        ax.set_xscale("log"); ax.set_xlabel("alpha · stronger regularization to the right")
        ax.set_ylabel("Mean CV MAPE (%)")
    else:
        selected = selected.sort_values("mean_cv_mape_pct")
        bars = ax.barh(range(len(selected)), selected.mean_cv_mape_pct, color="#777777")
        labels = selected.params.map(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()) or "No tuned parameter")
        ax.set_yticks(range(len(selected)), labels, fontsize=10)
        ax.invert_yaxis(); ax.set_xlabel("Mean CV MAPE (%)")
        ax.bar_label(bars, fmt="%.2f", padding=5)
        ax.set_xlim(0, selected.mean_cv_mape_pct.max() * 1.2)
    ax.set_title(f"Tuning: {winner['model']} · group {winner['feature_group']} · {winner['target_transform']} target", loc="left")
    ax.grid(alpha=.15)
    save(fig, dest, "05_selected_model_tuning.png")

    importance = pd.read_csv(output / "permutation_importance_valid.csv").sort_values("increase_mape_pp")
    fig, ax = plt.subplots(figsize=(10, max(4, .45 * len(importance) + 1.5)))
    ax.barh(importance.feature.map(FEATURE_LABELS), importance.increase_mape_pp, color="#69754C", height=.55)
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.set_xlabel("Mean increase in Valid MAPE after shuffling (percentage points)")
    ax.set_title("Feature contribution on 11 held-out cells", loc="left", pad=18)
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    save(fig, dest, "06_feature_contribution.png")

    # Compare group definitions without allowing Batch2/3 to choose a feature set.
    summary = cv[cv.feature_group.isin(["A", "B", "C", "D1", "D2"])].sort_values("mean_cv_mape_pct").drop_duplicates("feature_group")
    summary = summary.set_index("feature_group").loc[["A", "B", "C", "D1", "D2"]]
    summary.to_csv(output / "feature_group_comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 4.7))
    labels = ["A: core 5", "B: minimum Delta Q", "C: charging conditions", "D1: without capacity IQR", "D2: without capacity change"]
    bars = ax.barh(labels, summary.mean_cv_mape_pct, color="#777777", height=.55)
    ax.invert_yaxis(); ax.bar_label(bars, fmt="%.2f%%", padding=6)
    ax.set_xlim(0, summary.mean_cv_mape_pct.max() * 1.25)
    ax.set_xlabel("Best mean CV MAPE within each feature group (%)")
    ax.set_title("Feature design comparison on Batch1 development cells", loc="left", pad=18)
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    save(fig, dest, "07_feature_group_comparison.png")
    print(f"결과 그래프 7개 저장: {dest}", flush=True)
