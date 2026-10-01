"""원본 MATLAB v7.3 파일에서 셀 단위 피처를 재생성한다.

관측 종료 시점이나 100사이클 이후 측정값은 입력에 포함하지 않는다.
DAY1 캐시는 실행에 필요하지 않으며, 대조 검증에만 사용할 수 있다.
"""
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

BATCHES = {
    "Batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
SUMMARY_FIELDS = ["QDischarge", "QCharge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]
INPUT_FIELDS = ["deltaq_log10var", "deltaq_min", "qd_median", "qd_iqr",
                "qd_change100_10", "temp_mean", "charge_median",
                "policy_c1", "policy_switch_pct", "policy_c2"]


def numeric(dataset):
    if dataset.attrs.get("MATLAB_empty", 0):
        return np.empty(0, dtype=float)
    return np.asarray(dataset[()], dtype=float).ravel()


def text_value(dataset):
    return "".join(chr(int(x)) for x in dataset[()].ravel())


def initial_features(cycle, measurements, q10, q100, voltage, policy,
                     *, blank_first=False):
    """원래 사이클 번호 1-100만 사용. 빈 첫 기록은 근거 확인 후 결측화."""
    cycle = np.asarray(cycle)
    if (not np.all(np.isfinite(cycle)) or not np.all(cycle == np.floor(cycle))
            or len(np.unique(cycle)) != len(cycle)):
        raise ValueError("사이클 번호가 중복되거나 유효하지 않습니다.")
    early = (cycle >= 1) & (cycle <= 100)
    if set(cycle[early]) != set(range(1, 101)):
        raise ValueError("원래 번호의 초기 100사이클이 모두 필요합니다.")
    # Copy the window before cleaning: the caller's raw measurements are preserved.
    vals = {k: np.asarray(v, dtype=float)[early].copy() for k, v in measurements.items()}
    steps = cycle[early]
    if blank_first:
        first = steps == 1
        if not all(np.all(vals[k][first] == 0) for k in SUMMARY_FIELDS):
            raise ValueError("빈 첫 기록의 0값 근거와 일치하지 않습니다.")
        for value in vals.values():
            value[first] = np.nan
    qd = vals["QDischarge"]
    good = qd[np.isfinite(qd)]
    if len(good) != (99 if blank_first else 100):
        raise ValueError("설계에서 정하지 않은 초기 용량 결측입니다.")
    if (voltage.shape != q10.shape or q10.shape != q100.shape
            or len(voltage) != 1000 or not np.all(np.isfinite(voltage))
            or not np.all(np.diff(voltage) < 0)):
        raise ValueError("1000점 공통 방전 전압 격자를 확인하세요.")
    delta = q100 - q10
    if not np.all(np.isfinite(delta)):
        raise ValueError("ΔQ 곡선에 유효하지 않은 값이 있습니다.")
    variance = np.var(delta, ddof=1)
    if variance <= 0:
        raise ValueError("ΔQ 표본분산은 log10 변환을 위해 양수여야 합니다.")
    q25, q75 = np.quantile(good, [.25, .75], method="linear")
    row = {
        "valid_QD_early100": len(good), "deltaq_log10var": np.log10(variance),
        "deltaq_min": delta.min(), "qd_median": np.median(good),
        "qd_iqr": q75 - q25,
        "qd_change100_10": float(qd[steps == 100][0] - qd[steps == 10][0]),
        "temp_mean": np.nanmean(vals["Tavg"]),
        "charge_median": np.nanmedian(vals["chargetime"]),
        "policy_c1": np.nan, "policy_switch_pct": np.nan, "policy_c2": np.nan,
    }
    match = re.search(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C", policy)
    if match:
        row.update(policy_c1=float(match[1]), policy_switch_pct=float(match[2]),
                   policy_c2=float(match[3]))
    return row


def extract_features(data_dir):
    rows, audit, grid_reference = [], [], None
    for batch, filename in BATCHES.items():
        path = Path(data_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"데이터 파일이 필요합니다: {path}")
        print(f"원본 피처 생성: {batch}", flush=True)
        with h5py.File(path, "r") as f:
            refs = {k: f["batch"][k][()].ravel() for k in f["batch"]}
            for idx, ref in enumerate(refs["summary"]):
                cid = f"{batch}-C{idx:02d}"
                summary = f[ref]
                cycle = numeric(summary["cycle"])
                measurements = {k: numeric(summary[k]) for k in SUMMARY_FIELDS}
                if any(len(v) != len(cycle) for v in measurements.values()):
                    raise ValueError(f"{cid}: summary 필드 길이가 다릅니다.")
                curves = f[refs["cycles"][idx]]
                qrefs = curves["Qdlin"][()].ravel()
                def curve_at(n):
                    positions = np.flatnonzero(cycle == n)
                    if len(positions) != 1:
                        raise ValueError(f"{cid}: cycle {n}을 확인하세요.")
                    return numeric(f[qrefs[positions[0]]])
                first = cycle == 1
                zero = all(np.all(v[first] == 0) for v in measurements.values())
                empty_curve = curve_at(1).size == 0
                blank = batch == "Batch1" and zero and empty_curve
                if (batch == "Batch1" and not blank) or (batch != "Batch1" and zero):
                    raise ValueError(f"{cid}: 승인된 첫 기록 처리와 다릅니다.")
                voltage = numeric(f[refs["Vdlin"][idx]])
                if grid_reference is None:
                    grid_reference = voltage.copy()
                if not np.array_equal(voltage, grid_reference):
                    raise ValueError(f"{cid}: 셀 간 전압 격자가 다릅니다.")
                policy = text_value(f[refs["policy_readable"][idx]])
                life = numeric(f[refs["cycle_life"][idx]])
                if life.size != 1:
                    raise ValueError(f"{cid}: target 형식이 다릅니다.")
                available = bool(np.isfinite(life[0]) and life[0] > 0)
                row = {"cell_id": cid, "batch": batch, "policy": policy,
                       "cycle_life": life[0] if available else np.nan,
                       "target_available": available}
                row.update(initial_features(cycle, measurements, curve_at(10),
                                            curve_at(100), voltage, policy,
                                            blank_first=blank))
                rows.append(row)
                audit.append({"cell_id": cid, "blank_cycle1_marked_missing": blank,
                              "target_available": available,
                              "valid_QD_early100": row["valid_QD_early100"]})
    df = pd.DataFrame(rows)
    counts = df.groupby("batch").agg(cells=("cell_id", "size"),
                                      labelled=("target_available", "sum"))
    if counts.cells.tolist() != [46, 47, 46] or counts.labelled.tolist() != [46, 39, 44]:
        raise ValueError(f"DAY1의 데이터 구성과 다릅니다:\n{counts}")
    if not np.all(np.isfinite(df.loc[df.target_available, INPUT_FIELDS])):
        raise ValueError("학습·평가 대상의 입력에 결측이 있습니다. 임의 대치하지 않습니다.")
    return df, audit
