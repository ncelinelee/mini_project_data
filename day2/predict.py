"""저장된 모델로 셀 단위 피처 CSV를 예측한다. 정답 열은 필요하지 않다."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("day2/results/models/selected_model.joblib"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    rows = pd.read_csv(args.features)
    features = artifact["features"]
    if any(col not in rows for col in features):
        raise ValueError(f"입력 열이 필요합니다: {features}")
    if not np.all(np.isfinite(rows[features])):
        raise ValueError("모델 입력의 결측이나 무한대를 먼저 확인하세요.")
    output = rows[["cell_id"]].copy() if "cell_id" in rows else pd.DataFrame(index=rows.index)
    output["predicted_cycle_life"] = artifact["estimator"].predict(rows[features])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"{len(output)}개 셀 예측 저장: {args.output}")


if __name__ == "__main__":
    main()
