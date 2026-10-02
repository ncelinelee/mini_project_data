"""과제의 회귀 성능 표를 CV 평균과 고정 평가 결과에서 생성한다."""
import pandas as pd


def performance_tables(scores, winner, target=9.1):
    """MAPE의 악화 Gap은 후속 오차-기준 오차, 배치 비교는 B2-B3다."""
    values = scores.set_index("split").mape_pct
    train = float(winner["mean_cv_mape_pct"])
    valid = float(values["Valid"])
    test = float(values["Test_Batch2"])
    batch3 = float(values["Additional_Batch3"])
    gaps = {
        "train_batch1_cv_mape_pct": train,
        "valid_minus_train_cv_pp": valid - train,
        "valid_minus_fit_pp": valid - float(values["Fit_Batch1"]),
        "test_batch2_minus_valid_pp": test - valid,
        "test_batch2_minus_paper_pp": test - target,
        "batch2_minus_batch3_pp": test - batch3,
        "batch3_minus_paper_pp": batch3 - target,
        "paper_reference_mape_pct": target,
        "comparable_protocol": False,
        "gap_formulas": {
            "Gap (Train-Valid)": "Valid - Train (Batch 1 CV)",
            "Gap (Valid-Test)": "Test (Batch 2) - Valid",
            "Gap (Target-Test)": "Test - Target",
            "Gap (Batch2-Batch3)": "Test (Batch 2) - Test (Batch 3)",
        },
        "interpretation": "MAPE is percent; differences are percentage points. Positive B2-B3 means Batch2 error is larger.",
    }
    rows = [
        ("Train (Batch 1 CV)", train, "개발35셀·정책별5-fold MAPE의 산술평균"),
        ("Valid (Batch 1 Hold-out)", valid, "11셀·개발과 충전 정책이 겹치지 않는 hold-out"),
        ("Test (Batch 2)", test, "39셀·선정한 동일 모델로 최종 평가"),
        ("Gap (Train-Valid)", valid - train, "Valid−Train(CV), pp; (+) 내부 검증 악화"),
        ("Gap (Valid-Test)", test - valid, "Test(Batch2)−Valid, pp; (+) 배치 일반화 저하"),
        ("Gap (Target-Test)", test - target, f"Test(Batch2)−{target}%, pp; 과제 Target"),
    ]
    mandatory = pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])
    additional = mandatory.copy()
    additional["비교"] = additional["구분"].where(additional["구분"].str.startswith("Gap"), "")
    additional.loc[additional["비교"] != "", "구분"] = ""
    extra = pd.DataFrame([
        ("Test (Batch 3)", "", batch3, "44셀·같은 모델의 추가 평가"),
        ("", "Gap (Batch2-Batch3)", test - batch3, "Batch2−Batch3, pp; (+) Batch2 오차가 더 큼"),
        ("", "Gap (Target-Test)", batch3 - target, f"Test(Batch3)−{target}%, pp; 과제 공통 Target"),
    ], columns=["구분", "비교", "MAPE (%)", "비고"])
    additional = pd.concat([additional[extra.columns], extra], ignore_index=True)
    return mandatory, additional, gaps
