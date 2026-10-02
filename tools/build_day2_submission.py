"""실험 결과에서 제출 README와 실행 노트북을 작성한다."""
import json
import sys
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "day2/results"
sys.path.insert(0, str(ROOT))
from day2.reporting import performance_tables


def table(df, columns=None):
    if columns:
        df = df[columns]
    header = "| " + " | ".join(str(x) for x in df.columns) + " |\n"
    border = "| " + " | ".join("---" for _ in df.columns) + " |\n"
    lines = []
    for _, row in df.iterrows():
        values = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return header + border + "\n".join(lines)


def main():
    winner = json.loads((RESULTS / "selection.json").read_text())
    scores = pd.read_csv(RESULTS / "metrics.csv")
    cv = pd.read_csv(RESULTS / "cv_results.csv")
    champions = cv.sort_values("mean_cv_mape_pct").drop_duplicates("model")
    champions = champions[["model", "feature_group", "target_transform", "mean_cv_mape_pct", "std_cv_mape_pct", "params_json"]]
    baselines = cv[cv.model.isin(["Mean", "Univariate"])][[
        "model", "target_transform", "mean_cv_mape_pct", "std_cv_mape_pct"]]
    ablation = cv[(cv.model == winner["model"]) &
                  (cv.target_transform == winner["target_transform"]) &
                  (cv.params_json == winner["params_json"])][[
                      "feature_group", "mean_cv_mape_pct", "std_cv_mape_pct"]].sort_values("feature_group")
    ablation.to_csv(RESULTS / "matched_feature_comparison.csv", index=False)
    scores["split"] = scores["split"].replace({"Train": "Fit_Batch1"})
    mandatory, additional, gaps = performance_tables(scores, winner)
    mandatory.to_csv(RESULTS / "model_performance.csv", index=False)
    additional.to_csv(RESULTS / "model_performance_batch3.csv", index=False)
    score_table = table(additional)
    reflection = (ROOT / "docs/reflection.md").read_text().strip()
    readme = f'''# 초기 100사이클 기반 배터리 총수명 예측

DS Mini Project · 울산 캠퍼스 3반 · **U090 이나현**

초기 충·방전 관측으로 배터리의 총수명(`cycle_life`, 사이클)을 예측하는 회귀 실험이다. DAY1의 EDA를 근거로 피처·전처리·후보 모델·튜닝 범위를 고정하고, 충전 정책별 교차검증으로 후보를 선정했다. 딥러닝은 사용하지 않았다.

**선정 결과:** {winner['model']} · 피처군 {winner['feature_group']} · {winner['target_transform']}수명 · `alpha={winner['params'].get('alpha')}`, `l1_ratio={winner['params'].get('l1_ratio')}`. CV MAPE는 **{winner['mean_cv_mape_pct']:.2f}%**지만 Batch2 Test는 **{scores.set_index('split').loc['Test_Batch2','mape_pct']:.2f}%**다. 개발 교차검증의 성능이 다른 Batch에 그대로 일반화되지 않았다.

## 제출 자료

- [실행 결과가 포함된 분석 노트북](Day2_Battery_Cycle_Life.ipynb): EDA 해석 → 피처 생성 → 모델 비교·튜닝 → 최종 평가 → 오류·도메인 해석
- [모델·피처·하이퍼파라미터 설명](docs/modeling.md): 각 선택의 이유와 설정값의 역할
- [제출 조건·루브릭 대조](docs/submission_audit.md): 요구 내용과 실제 산출물의 위치, 확인된 한계
- [DAY1 설계 보고서](output/pdf/DS-MINI-Design-울산_3반-U090%20이나현.pdf)
- [과제 양식 성능 표](day2/results/model_performance.csv), [Batch3 추가 표](day2/results/model_performance_batch3.csv)
- [선정 결과](day2/results/selection.json), [전체 CV 비교](day2/results/cv_results.csv), [셀별 예측](day2/results/predictions.csv)

## 실행 환경과 재현 방법

검증한 환경은 Python 3.11.15, scikit-learn 1.9.1이다. CPU로 실행하며 설치 버전은 [requirements.txt](requirements.txt), 실행 환경·소스 해시는 [run_manifest.json](day2/results/run_manifest.json)에 기록했다.

```bash
git clone https://github.com/ncelinelee/mini_project_data.git
cd mini_project_data
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m day2.experiment --data-dir data-30 --jobs 4
python -m unittest discover -s tests -v
```

노트북은 프로젝트 루트에서 열고 위 가상환경을 커널로 선택한 뒤 전체 실행한다. 원본 피처를 재생성하고 탐색·평가까지 수행한다. CSV·JSON·그래프·저장 모델은 `day2/results/`에 생성된다. 병렬 프로세스 수는 계산 설정이며 비교 모델의 파라미터를 바꾸지 않는다.

단계를 따로 실행하려면 `--stage prepare`, `--stage tune`, `--stage evaluate`, `--stage plots`를 사용한다. 튜닝은 `prepare` 이후, 평가는 `tune` 이후 실행한다.

## 원본 데이터

[Kaggle MIT-Stanford 배터리 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle/data)를 내려받아 다음 파일을 `data-30/`에 둔다. 용량이 큰 원본 `.mat` 파일과 `.venv`는 Git에 포함하지 않는다.

```text
data-30/
  2017-05-12_batchdata_updated_struct_errorcorrect.mat
  2018-02-20_batchdata_updated_struct_errorcorrect.mat
  2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

`2018-04-03_varcharge`는 별도 충전 최적화 실험 데이터이므로 이번 학습·평가에서 제외한다. 원본은 HDF5 기반 MATLAB v7.3 형식이며 `h5py`로 필요한 배열만 읽는다.

| 데이터 | 관측 셀 | 수명 확인 셀 | 수명 중앙값 | <500 / >1000 셀 |
| --- | ---: | ---: | ---: | ---: |
| Batch1 | 46 | 46 | 858.5 | 0 / 10 |
| Batch2 | 47 | 39 | 472.0 | 28 / 3 |
| Batch3 | 46 | 44 | 1005.5 | 0 / 23 |

## EDA 결과와 구현 전략의 연결

| 질문 | 확인한 내용 | 구현에 반영한 선택 |
| --- | --- | --- |
| 수명 분포 | Batch2에 <500셀 28개, Batch1에는 0개. Batch3에 장수명 셀이 많음 | 회귀 선택, Batch별 평가·수명 구간별 오차 확인, 장수명 이상치 후보 유지 |
| 열화 곡선 | 초기 용량 상승·스파이크·이후 가속 등 단일 직선으로 설명하기 어려운 변화 | 용량 중앙값·IQR 비교. 전체 곡선의 knee는 EDA만 수행하고 입력 제외 |
| ΔQ(V) | Batch1 log분산과 수명 Pearson r=-0.886 | log분산 기본 피처, 한 변수 모델 기준, 최솟값 대체 비교 |
| 충전 조건 | 충전시간과 수명 Pearson r은 Batch1 +0.744, Batch2 -0.919, Batch3 +0.638 | 관계의 Batch 의존성 확인. 충전 조건은 C군에서 별도 비교, 정책 그룹 분리 |
| 피처 상관·중복 | IQR-변화량 r=-0.961/-0.408/-0.993. 평균·최고온도도 중복 | Ridge·ElasticNet 규제, D1·D2 제거 비교, 최고온도·IR 기본 입력 보류 |

수명 미확인 10셀은 관측 EDA에 유지하고 정답이 필요한 학습·평가에서 제외했다. Batch1의 46개 빈 첫 기록만 측정값을 결측으로 표시하고 원래 사이클 번호를 유지했다. 초기 유효 Qd는 Batch1 99개, Batch2·3 100개다. 스파이크는 제거하지 않았다.

## 피처·모델·튜닝

원래 번호 1-100의 측정만 사용하며 공통 1,000점 전압 격자에서 ΔQ=Q100(V)-Q10(V)를 계산한다. 표본분산의 `ddof=1`, IQR의 사분위수 계산은 선형 보간이다. 용량은 Ah, 온도는 °C, 충전시간은 분, 전환 기준은 %, 전류 조건은 C-rate다.

- A: ΔQ log분산·용량 중앙값·IQR·100-10 용량차·평균온도
- B: A의 log분산을 ΔQ 최솟값으로 대체
- C: A + 충전시간 중앙값·C1·전환%·C2
- D1: A에서 IQR 제외 / D2: A에서 용량차 제외

Mean은 피처 없는 기준선, Univariate는 강한 ΔQ 신호 한 개의 기준선이다. Ridge·ElasticNet은 소표본과 중복 피처, Forest·Boosting은 비선형 관계를 비교하기 위해 선정했다. 자연로그 타깃과 원수명 타깃을 각각 비교했다. 전체 파라미터 범위와 역할은 [모델 설명](docs/modeling.md)에 제시했다.

튜닝 조합 290개와 기준 후보 4개에 동일한 5-fold를 적용했다. 총 **1,470회 CV 학습** 중 튜닝 모델 학습은 1,450회다. 선형 입력의 표준화는 매 학습 폴드에서만 fit했다.

## 데이터 분리와 모델 선정

Batch1에서 정책 그룹의 20%를 seed=42로 보류했다. 개발 35셀·18정책 / Valid 11셀·5정책이며 [고정 셀 목록](day2/config/holdout.json)과 일치한다. 개발 데이터의 원문 정책 문자열로 `GroupKFold(5, shuffle=False)`를 구성했다. 같은 정책이 각 폴드의 학습·검증에 겹치지 않는다.

선정 점수는 **5개 폴드 MAPE(%)의 산술평균**이다. 표본 수와 관계없이 각 폴드 가중치는 1/5다. 반올림 전 최솟값으로 모델·피처군·타깃 변환·파라미터를 선택했다. pooled OOF MAPE는 {winner['pooled_oof_mape_pct']:.3f}%로 별도 값이며 선정 기준이 아니다.

선정 결과를 저장한 뒤 개발 35셀 전체로 최종 모델을 학습했다. Valid·Batch2·Batch3 결과로 재튜닝하거나 피처를 바꾸지 않았다. [CV 분할 기록](day2/results/cv_splits.json)과 [전체 경고 기록](day2/results/fit_warnings.json)을 보존했다. 이번 실행의 수렴 경고는 0건이다.

## 모델 비교 결과

모델별 최저 CV MAPE는 다음과 같다. 모델·피처군·목표 변환을 함께 선정한 결과다. 설정값은 [전체 비교](day2/results/cv_results.csv)와 노트북에서 확인할 수 있다.

{table(champions[['model', 'feature_group', 'target_transform', 'mean_cv_mape_pct']])}

![모델별 CV MAPE](day2/results/figures/01_model_comparison.png)

ElasticNet·D1·ln수명을 탐색 범위의 최저 CV 평균으로 선정했다. 같은 ElasticNet·ln수명·alpha=0.01·l1_ratio=0.5에서 IQR 제외 D1은 6.230%, 기본 A는 6.755%였다. 중복 피처 제거의 이득을 확인했지만 독립성이나 인과관계를 입증한 결과는 아니다. [동일 설정의 피처 비교](day2/results/matched_feature_comparison.csv)

최종 입력은 ΔQ log분산·용량 중앙값·100-10 용량차·평균온도다. `alpha=0.01`은 전체 규제 강도, `l1_ratio=0.5`는 L1·L2 규제 비중이다. 고정 설정은 `max_iter=10000`, `tol=1e-4`, `selection='cyclic'`, `fit_intercept=True`다. 학습 폴드에서 입력을 표준화했다. 평균온도 계수는 0이지만 평가 후 피처군을 바꾸지 않았다. 모든 후보의 설정값·역할은 [모델 설명](docs/modeling.md)에 정리했다.

## 최종 성능

과제의 **Train은 Batch1 개발 35셀의 정책별 5-fold MAPE 산술평균**이다. Valid는 독립 hold-out 11셀이다. MAPE는 %, Gap은 pp이며 아래 표에서 Gap 행의 단위를 구분한다. MAE·RMSE와 학습 셀 재예측 오차(Fit 5.858%)는 [보조 지표](day2/results/metrics.csv)에 별도로 보존한다.

{score_table}

![실제 수명과 예측 수명](day2/results/figures/02_actual_vs_predicted.png)

과제의 행 이름을 유지하되 (+)가 악화를 나타내도록 **Train-Valid 행은 Valid−Train(CV)**, Valid-Test 행은 Test(Batch2)−Valid, Target-Test 행은 해당 Test−9.1%로 계산했다. 내부 Gap {gaps['valid_minus_train_cv_pp']:.2f}pp는 소표본·정책 구성과 후보 선정의 낙관성을 포함할 수 있어 과적합의 단독 증거로 보지 않는다. Batch2-Batch3는 B2−B3={gaps['batch2_minus_batch3_pp']:.2f}pp로, Batch2 오차가 더 크다는 뜻이다.

## 오류 분석과 Batch 차이

![수명 구간별 오차](day2/results/figures/04_lifetime_group_errors.png)

Batch2의 <500사이클 28셀 MAPE는 45.56%이며 평균적으로 수명을 약 204사이클 과대 예측했다. 개발 데이터에는 <500셀과 같은 짧은 수명 영역이 없었다. Batch2에서는 용량 중앙값 21/39셀, ΔQ log분산 11/39셀이 개발 입력의 최댓값을 넘었다. 수명·입력 분포 차이는 일반화 실패를 해석하는 근거이며 오차의 단일 인과 원인으로 단정하지 않는다. 상세 수치는 [입력 범위 비교](day2/results/feature_range_shift.csv)에 기록했다.

Valid의 >1000사이클 3셀 MAPE는 34.60%로 과대 예측이 두드러졌다. 반대로 Batch3의 >1000사이클 23셀 MAPE는 18.01%이며 평균 약 247사이클 과소 예측했다. 특히 C07·C38·C45(1836·1935·1801사이클)의 오차는 38.76·46.54·35.76%였다. 장수명 셀을 제거하지 않고 평가에 포함했다.

![수명에 따른 셀별 오차](day2/results/figures/03_error_by_lifetime.png)

![검증 셀에서의 피처 기여](day2/results/figures/06_feature_contribution.png)

Valid 11셀에서 피처를 30회 섞었을 때 ΔQ log분산의 평균 MAPE 증가가 가장 컸다. 용량차의 증가값은 약 -0.10pp로 작은 음수였다. 이 검증 표본에서 기여가 명확하지 않다는 의미이며 피처가 유해하다는 인과적 결론은 아니다. 평가 후 모델 변경에는 사용하지 않았다.

Batch3는 전체 MAPE가 Batch2보다 낮지만 장수명 구간의 오차가 남는다. 수명 분포와 MAPE의 분모가 다르므로 전체 평균만으로 특정 배치에 대한 과적합을 단정하지 않는다. 공통 전압 격자에서 같은 셀의 ΔQ를 계산했지만, 배치별 곡선 시작점 차이를 물리적으로 정렬·보정했는지는 검증하지 않았다. 스파이크·장수명 셀은 기록 오류 근거가 없어 보존했다. [추가 검증의 한계](docs/modeling.md#batch3의-추가-검증-한계)

## 논문 성능과 비교

과제에서 제시한 원논문 회귀 성능 기준은 MAPE 9.1%다. 본 Batch2 Test는 40.81%로 참고값보다 {gaps['test_batch2_minus_paper_pp']:.2f}pp 크고, Batch3 추가 평가는 12.65%로 {gaps['batch3_minus_paper_pp']:.2f}pp 크다. CV 6.23%만으로 논문보다 우수하다고 판단할 수 없다. [원논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)

논문의 정제된 124셀(학습41·1차 테스트43·2차 테스트40)과 달리 이번 실험은 원본 139셀 중 수명 확인 129셀을 사용하며 정책 그룹으로 분리했다. 저자 코드의 연속 실험 연결·제외 처리까지 동일하게 재현한 실험이 아니다. DAY1 EDA에서 Valid·Batch2·Batch3 수명을 이미 확인했으므로 완전히 미관측한 테스트라고 주장하지 않는다. 비교값은 조건 차이를 명시한 참고다.

{reflection}

## ESS 활용과 한계

초기 셀 관측으로 수명 위험을 선별하고 추가 점검의 우선순위를 정하는 연구용 보조 신호로 검토할 수 있다. 다만 데이터는 실험실의 LFP/graphite 셀과 고속 충전 조건에서 얻은 자료다. 실제 ESS의 충·방전 부하, 온도 변화, 휴지·달력 열화, 셀 편차와 팩 운용을 대표하지 않는다. [실험 조건](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)

이번 Batch2 과대 예측은 수명이 짧은 셀의 위험을 낮게 평가할 수 있다는 한계를 보여준다. 실제 교체 시점이나 안전 제어에 바로 적용할 근거는 부족하다. 적용하려면 별도의 ESS 운용 데이터와 외부 검증, 수명 정의 일치, 입력 범위 점검이 필요하다. 향후 다른 데이터로 개선을 검토하더라도 이번 평가셋으로 설정을 다시 고른 결과와 혼합하지 않아야 한다.

## 검증 및 역할

- 원본 재생성 피처와 DAY1 피처 수치가 일치한다.
- 원래 cycle100 사용, 미래 측정값 미사용, 원본 보존, 로그 역변환, 학습 폴드 표준화, 정책 분리 계약을 테스트했다.
- 저장 모델을 다시 불러온 예측이 원래 예측과 일치한다.
- 전체 실행 노트북의 재현 확인은 [verification.json](day2/results/verification.json)에 기록한다.
- 담당: U090 이나현 — EDA 해석, 피처·모델 전략 결정, 결과 검토 및 제출.

## 파일 구조

```text
Day2_Battery_Cycle_Life.ipynb  # 실행 결과·그래프·해석
day2/data.py                 # 원본에서 초기 피처 생성
day2/modeling.py             # 분리·모델·Grid·점수
day2/experiment.py           # 추출 → 튜닝 → 최종 평가
day2/reporting.py            # 과제 양식 성능 표·Gap
day2/plots.py                # 결과 그래프
day2/predict.py              # 저장 모델로 피처 CSV 예측
day2/config/                 # DAY1의 고정 설계와 셀 분리
day2/results/                # 실험 결과·예측·그래프·모델
docs/modeling.md             # 선택 이유와 파라미터 역할
tests/test_day2_contract.py  # 데이터 누출·분리·점수 계약 검증
```

저장 모델로 정답 없이 예측하려면 같은 단위로 계산한 선택 피처 CSV를 사용한다.

```bash
python -m day2.predict --features input_features.csv --output predictions.csv
```

## 참고문헌

- [Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)
- [저자 공개 코드와 데이터 처리](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [Kaggle 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle/data)
- [scikit-learn 공식 문서와 모델별 참고](docs/modeling.md#참고)
'''
    (ROOT / "README.md").write_text(readme)
    cells = []
    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbformat.v4.new_code_cell(text.strip()))
    def show(path):
        code(f"display(Image(filename=str(ROOT / {path!r}), width=1000))")

    md("""# 초기 100사이클 기반 배터리 총수명 예측

**DS Mini Project · 울산 캠퍼스 3반 · U090 이나현**

DAY1 설계에 따라 총수명 회귀를 구현한다. 원본에서 입력을 다시 생성하고 정책별 교차검증으로 모델·피처·타깃 변환·하이퍼파라미터를 비교한다. 그래프마다 확인한 내용과 모델 설계 또는 결과 해석을 연결한다.

원본 3개 `.mat` 파일은 README의 경로에 준비하고 `requirements.txt`를 설치한 Python 환경을 선택한다. 전체 실행은 1,470회 CV 학습과 최종 평가를 수행한다.""")
    code("""from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = Path.cwd().resolve()
while not (ROOT / 'day2/config/design.json').exists():
    if ROOT == ROOT.parent:
        raise FileNotFoundError('프로젝트 폴더에서 노트북을 실행하세요.')
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'day2/results'
from day2.experiment import prepare, tune, evaluate, load_design
from day2.plots import make_figures
design, holdout = load_design(ROOT)
print('Python:', sys.version.split()[0])
print('문제:', design['task'], '| 목표:', design['target'])""")
    md("""## 1. EDA에서 확인한 데이터 특성

Batch1·2·3 모두 관측 EDA의 대상이다. 수명이 없는 10셀은 EDA에 남기고 학습·평가에서 제외한다. Batch1의 빈 첫 기록은 결측으로 표시하되 원래 사이클 번호는 유지한다. 스파이크와 장수명 이상치 후보는 제거하지 않는다.

### Q1. 수명 분포 → 회귀와 Batch별 평가

수명 중앙값은 Batch1 858.5, Batch2 472, Batch3 1005.5사이클이다. <500셀은 각각 0·28·0개로 큰 차이가 있다. Batch1의 550사이클 분류는 1:45로 불균형해 회귀를 선택했다. Batch2의 짧은 수명과 Batch3의 장수명에 대한 오류를 별도로 확인한다.""")
    show("day1/readable_figures/01_life_histograms.png")
    md("""### Q2. 열화 곡선 → 강건한 요약과 미래 정보 제외

초기 Qd에는 용량 상승과 스파이크가 있고 전체 곡선에서는 이후 열화가 가속될 수 있다. 평균·표준편차만으로 요약하지 않고 중앙값·IQR을 비교한다. 아래 그림은 Batch별 중앙값과 사분위 범위다. 전체 수명 곡선으로 찾은 knee는 설명용 EDA이며 미래 정보라 모델 입력에 넣지 않는다.""")
    show("day1/readable_figures/06_capacity_early100_zoom.png")
    md("""### Q3. ΔQ(V) → log분산 기본 피처와 한 변수 기준

ΔQ는 원래 cycle100의 Q(V)에서 cycle10의 Q(V)를 뺀 값이다. Batch1의 log분산과 수명 Pearson r=-0.886으로 강한 음의 관계를 보였다. 이 신호의 단독 설명력을 한 변수 모델로 확인하고, 최솟값 대체군 B도 비교한다.""")
    show("day1/readable_figures/11_deltaq_variance_scatter.png")
    md("""### Q4. 충전 조건 → 별도 피처군과 정책별 분리

충전시간과 수명의 Pearson r은 Batch1 +0.744, Batch2 -0.919, Batch3 +0.638로 방향이 다르다. 하나의 공통 인과관계로 해석하지 않는다. 충전 조건을 C군에서 별도 비교하며, 같은 정책이 학습·검증에 중복되지 않도록 정책 그룹으로 분리한다. 전류 평균/RMS와 초기 열화 기울기 분석은 EDA에만 사용한다.""")
    show("day1/readable_figures/13_charge_time_life.png")
    md("""### Q5. 상관·중복 → 규제와 제거 비교

IQR과 100-10 용량차의 Pearson r은 Batch1 -0.961, Batch2 -0.408, Batch3 -0.993이다. 독립성을 확보했다고 주장하지 않는다. 기본 A와 IQR 제외 D1·변화량 제외 D2를 동일한 CV에서 비교하고 Ridge·ElasticNet으로 계수 변동을 줄인다. 평균·최고온도의 중복 때문에 평균온도를 기본 후보로 선택했다. IR은 0값 의미 확인 전 보류했다.""")
    md("""## 2. 원본에서 피처 생성

미래 측정값과 관측 종료 시점은 입력에 포함하지 않는다. 공통 전압 격자의 차이만 계산하며 추가 보간하지 않는다. 실제 스파이크 값은 유지한다. 학습·평가에 필요한 모든 피처가 유효한지 확인하고 설계 밖 결측에는 자동 대치하지 않는다.""")
    code("""features = prepare(ROOT / 'data-30', OUTPUT)
display(features.groupby('batch').agg(관측셀=('cell_id','size'), 수명확인=('target_available','sum')))
display(features[['cell_id', 'batch', 'target_available', 'valid_QD_early100']].head())""")
    md("""## 3. 피처·모델·튜닝 설계

A 5개, B ΔQ 최솟값 대체, C 충전 조건 추가, D1 IQR 제외, D2 용량차 제외를 비교한다. 원수명과 ln수명을 각각 학습하며 로그 예측은 exp로 되돌린 뒤 평가한다. 선형 입력은 학습 폴드에서만 표준화한다.

Mean은 단순 기준선, 한 변수 선형 모델은 ΔQ의 설명력, Ridge·ElasticNet은 소표본과 중복 피처, Forest·Boosting은 비선형 관계를 확인한다. 각 후보 선정 이유와 조정·고정 파라미터의 역할은 아래에 명시한다.""")
    model_docs = (ROOT / "docs/modeling.md").read_text()
    md(model_docs[model_docs.index("## 모델 후보와 EDA의 연결"):model_docs.index("## 선정과 평가 절차")])
    code("""display(pd.DataFrame([{'피처군': g, '입력수': len(fs), '피처': ', '.join(fs)}
                      for g, fs in design['feature_groups'].items()]))
print('개발:', len(holdout['development_cell_ids']), '셀')
print('별도 검증:', len(holdout['holdout_cell_ids']), '셀')""")
    md("""## 4. 정책별 Grid Search와 모델 선정

개발 35셀의 정책별 5-fold를 모든 후보에 동일하게 적용한다. 정답을 원래 단위로 예측한 5개 폴드 MAPE의 산술평균이 주 지표다. 각 가중치는 1/5이며 반올림 전 최솟값으로 선정한다. Batch2·3은 후보 점수 계산에 사용하지 않는다. 290개 튜닝 조합과 기준 후보 4개로 총 1,470회 학습한다.""")
    code("""winner = tune(features, OUTPUT, jobs=4)
cv = pd.read_csv(OUTPUT / 'cv_results.csv')
champions = cv.sort_values('mean_cv_mape_pct').drop_duplicates('model')
display(champions[['model','feature_group','target_transform','mean_cv_mape_pct','std_cv_mape_pct','params_json']])
display(pd.DataFrame(winner['fold_results']))
display(Markdown(f\"**선정:** {winner['model']} / {winner['feature_group']} / {winner['target_transform']} / {winner['params']}\\n\\nCV MAPE {winner['mean_cv_mape_pct']:.3f}% · pooled OOF {winner['pooled_oof_mape_pct']:.3f}%\"))""")
    md("""## 5. 고정한 모델의 최종 평가

선정 결과를 저장한 뒤 개발 35셀로 최종 fit한다. 제출 표의 Train은 정책별 5-fold MAPE 산술평균이며, Valid는 별도 Batch1 hold-out 11셀이다. 학습 셀 재예측은 Fit_Batch1 보조 지표로 분리한다. Batch2 39셀과 Batch3 44셀에도 같은 모델을 적용한다. 평가 점수로 설정을 바꾸지 않는다.

과제의 Gap 이름을 유지하되 Train-Valid=Valid−Train(CV), Valid-Test=Test(Batch2)−Valid, Target-Test=해당 Test−9.1%로 계산한다. Batch2-Batch3=Test(Batch2)−Test(Batch3)다. 모든 Gap의 단위는 pp다.""")
    code("""scores = evaluate(features, OUTPUT)
make_figures(OUTPUT)
display(pd.read_csv(OUTPUT / 'model_performance_batch3.csv').fillna(''))
display(Markdown('**보조 지표:** Fit_Batch1은 학습 셀 재예측 오차이며 Train(CV)과 다르다. MAE·RMSE 단위는 사이클이다.'))
display(scores)
display(pd.read_csv(OUTPUT / 'feature_group_comparison.csv'))""")
    for path in ["01_model_comparison.png", "07_feature_group_comparison.png", "05_selected_model_tuning.png"]:
        code(f"display(Image(filename=str(OUTPUT / 'figures/{path}'), width=1000))")
    md("""**해석:** ElasticNet·D1·ln수명이 비교 범위에서 가장 낮은 CV 평균 MAPE를 얻었다. IQR을 제외한 조합이 유리했으며 규제와 중복 피처 검증이라는 설계가 구현에 연결됐다. alpha=0.01, l1_ratio=0.5가 선정됐다. 온도 계수는 0이지만 입력군을 평가 후 다시 바꾸지 않는다. CV 최솟값은 탐색한 후보 중 결과이며 전역 최적이나 외부 성능을 보장하지 않는다.""")
    md("### 원수명·로그 기준선과 같은 설정의 피처 비교\n\n"
       "원수명 평균 기준선과 로그 역변환 기준선은 각각 계산한다. 피처군별 최저점 비교는 모델 설정도 다를 수 있으므로, 같은 ElasticNet·ln수명·alpha=0.01·l1_ratio=0.5의 결과도 대조한다.\n\n"
       + table(baselines) + "\n\n" + table(ablation)
       + "\n\nD1과 A의 차이는 약 0.526pp다. 동일한 설정에서 IQR 제외 조합이 낮은 오차를 보였으며, 외부 성능 개선을 보장하지 않는다.")
    code("display(Image(filename=str(OUTPUT / 'figures/02_actual_vs_predicted.png'), width=1000))")
    md("""## 6. 오류 분석과 일반화 한계

CV 6.23%, Valid 14.97%, Batch2 40.81%, Batch3 12.65%로 평가 조건에 따라 차이가 크다. 특히 Batch2의 단수명 셀을 과대 예측했다. 개발 학습에서 짧은 수명과 해당 입력 범위를 충분히 관측하지 못한 점을 함께 확인한다. 이 차이를 특정 피처 하나의 인과효과로 단정하지 않는다.""")
    code("""display(pd.read_csv(OUTPUT / 'subgroup_metrics.csv'))
display(pd.read_csv(OUTPUT / 'feature_range_shift.csv'))
display(pd.read_csv(OUTPUT / 'long_life_cells.csv'))
display(pd.read_csv(OUTPUT / 'largest_errors.csv').head(5))""")
    code("display(Image(filename=str(OUTPUT / 'figures/04_lifetime_group_errors.png'), width=1000))")
    md("""**해석:** Batch2 <500사이클 28셀의 MAPE 45.56%, 평균 과대 예측 약 204사이클이다. Batch3 >1000사이클 23셀은 MAPE 18.01%, 평균 과소 예측 약 247사이클이다. C07·C38·C45는 35.76-46.54%의 오차를 보였다. 장수명 이상치 후보는 실제 평가에 남겼다. 공통 전압 격자에서 같은 셀의 ΔQ를 계산했지만 배치별 곡선 시작점의 물리적 정렬·보정은 검증하지 않았다. 이는 Batch3 추가 평가의 한계이며 다음 실험에서 원시 곡선 조건을 확인해야 한다.""")
    code("display(Image(filename=str(OUTPUT / 'figures/03_error_by_lifetime.png'), width=1000))")
    md("""### 피처 기여와 해석

Valid 입력을 30회 섞어 MAPE 변화를 확인한다. 모델 재선정에는 사용하지 않는다. 중복 피처의 중요도는 분산될 수 있으며 작은 음수는 이 표본에서 기여가 명확하지 않다는 의미다. 선형 계수는 표준화 입력과 ln수명 사이의 값으로, 사이클 증가량과 동일하게 읽지 않는다.""")
    code("""display(pd.read_csv(OUTPUT / 'permutation_importance_valid.csv'))
if (OUTPUT / 'linear_coefficients.csv').exists():
    display(pd.read_csv(OUTPUT / 'linear_coefficients.csv'))
display(Image(filename=str(OUTPUT / 'figures/06_feature_contribution.png'), width=1000))""")
    md("""## 7. 논문 성능 비교

초기 100사이클을 이용한 논문의 참고 테스트 오차는 9.1%다. 본 Batch2는 +31.71pp, Batch3는 과제의 공통 9.1% 기준으로 +3.55pp다. 내부 Valid−Train(CV)은 +8.74pp, Batch2−Valid는 +25.85pp다. Batch2−Batch3는 +28.16pp로 Batch2의 오차가 더 크다. 수명 분포와 MAPE의 분모가 달라 이 차이만으로 특정 배치 과적합을 단정하지 않는다. CV만으로 논문보다 우수하다고 주장할 수 없다.

논문은 정제된 124셀(41/43/40), 본 실험은 원본 139셀 중 수명 확인 129셀과 정책별 분리를 사용한다. 저자의 연속 실험 연결과 제외까지 동일한 재현이 아니다. DAY1에서 평가 대상 수명을 이미 EDA로 확인한 점도 한계다. [원논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)""")
    code("display(pd.Series(json.loads((OUTPUT / 'performance_gaps.json').read_text())))")
    md(reflection.replace("## 실험을 통해 배운 점과 다음에 확인할 내용", "## 8. 실험을 통해 배운 점과 다음에 확인할 내용", 1))
    md("""## 9. ESS 활용과 한계

초기 수명 위험의 선별과 추가 점검 우선순위를 정하는 보조 신호로 검토할 수 있다. 다만 실험실의 LFP/graphite 셀과 고속 충전 조건은 실제 ESS의 부하·온도·달력 열화·팩 운용을 대표하지 않는다. Batch2 과대 예측은 단수명 위험을 낮게 평가할 수 있어 직접적인 교체·안전 제어에 적용할 근거가 부족하다.

별도의 ESS 운용 데이터, 수명 정의 일치, 입력 범위 점검과 외부 검증이 필요하다. 이번 평가값으로 재튜닝한 결과를 기존 테스트와 섞지 않는다. [실험 조건](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)

## 10. 재현 기록과 산출물

설정·분할·예측·모델·환경을 저장했다. 모델을 재로딩한 예측 일치와 선정 파일이 평가 중 바뀌지 않았음을 확인한다. 데이터 누출 관련 계약 검증은 `python -m unittest discover -s tests -v`로 실행한다.""")
    code("""manifest = json.loads((OUTPUT / 'run_manifest.json').read_text())
display(pd.Series({k: manifest[k] for k in ['python', 'model_reload_prediction_check',
    'selection_unchanged_during_evaluation', 'convergence_warning_count_cv', 'negative_prediction_count']}))
print('결과 파일:', ', '.join(p.name for p in sorted(OUTPUT.glob('*.csv'))))""")
    md("""## 참고문헌과 역할

- Severson et al. (2019), [Nature Energy 논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)
- [저자 공개 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [Kaggle 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle/data)
- [scikit-learn 모델·검증 문서](docs/modeling.md#참고)

U090 이나현: EDA 해석, 피처·모델 전략 결정, 결과 검토 및 제출.""")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.15"}})
    nbformat.write(notebook, ROOT / "Day2_Battery_Cycle_Life.ipynb")
    print("README와 실행 노트북 작성 완료")


if __name__ == "__main__":
    main()
