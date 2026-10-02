# 초기 100사이클 기반 배터리 총수명 예측

DS Mini Project · 울산 캠퍼스 3반 · **U090 이나현**

초기 충·방전 관측으로 배터리의 총수명(`cycle_life`, 사이클)을 예측하는 회귀 실험이다. DAY1의 EDA를 근거로 피처·전처리·후보 모델·튜닝 범위를 고정하고, 충전 정책별 교차검증으로 후보를 선정했다. 딥러닝은 사용하지 않았다.

**선정 결과:** ElasticNet · 피처군 D1 · ln수명 · `alpha=0.01`, `l1_ratio=0.5`. CV MAPE는 **6.23%**지만 Batch2 Test는 **40.81%**다. 개발 교차검증의 성능이 다른 Batch에 그대로 일반화되지 않았다.

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

선정 점수는 **5개 폴드 MAPE(%)의 산술평균**이다. 표본 수와 관계없이 각 폴드 가중치는 1/5다. 반올림 전 최솟값으로 모델·피처군·타깃 변환·파라미터를 선택했다. pooled OOF MAPE는 6.177%로 별도 값이며 선정 기준이 아니다.

선정 결과를 저장한 뒤 개발 35셀 전체로 최종 모델을 학습했다. Valid·Batch2·Batch3 결과로 재튜닝하거나 피처를 바꾸지 않았다. [CV 분할 기록](day2/results/cv_splits.json)과 [전체 경고 기록](day2/results/fit_warnings.json)을 보존했다. 이번 실행의 수렴 경고는 0건이다.

## 모델 비교 결과

모델별 최저 CV MAPE는 다음과 같다. 모델·피처군·목표 변환을 함께 선정한 결과다. 설정값은 [전체 비교](day2/results/cv_results.csv)와 노트북에서 확인할 수 있다.

| model | feature_group | target_transform | mean_cv_mape_pct |
| --- | --- | --- | --- |
| ElasticNet | D1 | ln | 6.230 |
| Ridge | D1 | identity | 6.493 |
| Univariate | univariate | ln | 7.134 |
| GradientBoosting | D2 | ln | 9.072 |
| RandomForest | D2 | ln | 9.430 |
| Mean | baseline | identity | 19.006 |

![모델별 CV MAPE](day2/results/figures/01_model_comparison.png)

ElasticNet·D1·ln수명을 탐색 범위의 최저 CV 평균으로 선정했다. 같은 ElasticNet·ln수명·alpha=0.01·l1_ratio=0.5에서 IQR 제외 D1은 6.230%, 기본 A는 6.755%였다. 중복 피처 제거의 이득을 확인했지만 독립성이나 인과관계를 입증한 결과는 아니다. [동일 설정의 피처 비교](day2/results/matched_feature_comparison.csv)

최종 입력은 ΔQ log분산·용량 중앙값·100-10 용량차·평균온도다. `alpha=0.01`은 전체 규제 강도, `l1_ratio=0.5`는 L1·L2 규제 비중이다. 고정 설정은 `max_iter=10000`, `tol=1e-4`, `selection='cyclic'`, `fit_intercept=True`다. 학습 폴드에서 입력을 표준화했다. 평균온도 계수는 0이지만 평가 후 피처군을 바꾸지 않았다. 모든 후보의 설정값·역할은 [모델 설명](docs/modeling.md)에 정리했다.

## 최종 성능

과제의 **Train은 Batch1 개발 35셀의 정책별 5-fold MAPE 산술평균**이다. Valid는 독립 hold-out 11셀이다. MAPE는 %, Gap은 pp이며 아래 표에서 Gap 행의 단위를 구분한다. MAE·RMSE와 학습 셀 재예측 오차(Fit 5.858%)는 [보조 지표](day2/results/metrics.csv)에 별도로 보존한다.

| 구분 | 비교 | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| Train (Batch 1 CV) |  | 6.230 | 개발35셀·정책별5-fold MAPE의 산술평균 |
| Valid (Batch 1 Hold-out) |  | 14.965 | 11셀·개발과 충전 정책이 겹치지 않는 hold-out |
| Test (Batch 2) |  | 40.813 | 39셀·선정한 동일 모델로 최종 평가 |
|  | Gap (Train-Valid) | 8.735 | Valid−Train(CV), pp; (+) 내부 검증 악화 |
|  | Gap (Valid-Test) | 25.848 | Test(Batch2)−Valid, pp; (+) 배치 일반화 저하 |
|  | Gap (Target-Test) | 31.713 | Test(Batch2)−9.1%, pp; 과제 Target |
| Test (Batch 3) |  | 12.652 | 44셀·같은 모델의 추가 평가 |
|  | Gap (Batch2-Batch3) | 28.161 | Batch2−Batch3, pp; (+) Batch2 오차가 더 큼 |
|  | Gap (Target-Test) | 3.552 | Test(Batch3)−9.1%, pp; 과제 공통 Target |

![실제 수명과 예측 수명](day2/results/figures/02_actual_vs_predicted.png)

과제의 행 이름을 유지하되 (+)가 악화를 나타내도록 **Train-Valid 행은 Valid−Train(CV)**, Valid-Test 행은 Test(Batch2)−Valid, Target-Test 행은 해당 Test−9.1%로 계산했다. 내부 Gap 8.74pp는 소표본·정책 구성과 후보 선정의 낙관성을 포함할 수 있어 과적합의 단독 증거로 보지 않는다. Batch2-Batch3는 B2−B3=28.16pp로, Batch2 오차가 더 크다는 뜻이다.

## 오류 분석과 Batch 차이

![수명 구간별 오차](day2/results/figures/04_lifetime_group_errors.png)

Batch2의 <500사이클 28셀 MAPE는 45.56%이며 평균적으로 수명을 약 204사이클 과대 예측했다. 개발 데이터에는 <500셀과 같은 짧은 수명 영역이 없었다. Batch2에서는 용량 중앙값 21/39셀, ΔQ log분산 11/39셀이 개발 입력의 최댓값을 넘었다. 수명·입력 분포 차이는 일반화 실패를 해석하는 근거이며 오차의 단일 인과 원인으로 단정하지 않는다. 상세 수치는 [입력 범위 비교](day2/results/feature_range_shift.csv)에 기록했다.

Valid의 >1000사이클 3셀 MAPE는 34.60%로 과대 예측이 두드러졌다. 반대로 Batch3의 >1000사이클 23셀 MAPE는 18.01%이며 평균 약 247사이클 과소 예측했다. 특히 C07·C38·C45(1836·1935·1801사이클)의 오차는 38.76·46.54·35.76%였다. 장수명 셀을 제거하지 않고 평가에 포함했다.

![수명에 따른 셀별 오차](day2/results/figures/03_error_by_lifetime.png)

![검증 셀에서의 피처 기여](day2/results/figures/06_feature_contribution.png)

Valid 11셀에서 피처를 30회 섞었을 때 ΔQ log분산의 평균 MAPE 증가가 가장 컸다. 용량차의 증가값은 약 -0.10pp로 작은 음수였다. 이 검증 표본에서 기여가 명확하지 않다는 의미이며 피처가 유해하다는 인과적 결론은 아니다. 평가 후 모델 변경에는 사용하지 않았다.

Batch3는 전체 MAPE가 Batch2보다 낮지만 장수명 구간의 오차가 남는다. 수명 분포와 MAPE의 분모가 다르므로 전체 평균만으로 특정 배치에 대한 과적합을 단정하지 않는다. 공통 전압 격자에서 같은 셀의 ΔQ를 계산했지만, 배치별 곡선 시작점 차이를 물리적으로 정렬·보정했는지는 검증하지 않았다. 스파이크·장수명 셀은 기록 오류 근거가 없어 보존했다. [추가 검증의 한계](docs/modeling.md#batch3의-추가-검증-한계)

## 논문 성능과 비교

과제에서 제시한 원논문 회귀 성능 기준은 MAPE 9.1%다. 본 Batch2 Test는 40.81%로 참고값보다 31.71pp 크고, Batch3 추가 평가는 12.65%로 3.55pp 크다. CV 6.23%만으로 논문보다 우수하다고 판단할 수 없다. [원논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)

논문의 정제된 124셀(학습41·1차 테스트43·2차 테스트40)과 달리 이번 실험은 원본 139셀 중 수명 확인 129셀을 사용하며 정책 그룹으로 분리했다. 저자 코드의 연속 실험 연결·제외 처리까지 동일하게 재현한 실험이 아니다. DAY1 EDA에서 Valid·Batch2·Batch3 수명을 이미 확인했으므로 완전히 미관측한 테스트라고 주장하지 않는다. 비교값은 조건 차이를 명시한 참고다.

## 실험을 통해 배운 점과 다음에 확인할 내용

DAY1에서 ΔQ log분산과 수명의 관계가 강하게 나타나 이 신호를 중심으로 회귀 모델을 설계했다. 용량 피처의 중복을 줄이는 비교와 규제 모델, 비선형 모델을 같은 정책별 교차검증에 적용했다. 그 결과 IQR을 제외한 D1군의 ElasticNet이 선정됐다. 원본 피처 재생성, 학습 폴드에서만 수행하는 표준화, 로그 예측의 역변환, 고정된 셀 분리를 코드로 연결했고 전체 실행에서도 같은 결과가 재현됐다. 이 과정을 통해 EDA에서 정한 선택을 실제 실험으로 확인할 수 있었다.

CV MAPE는 6.23%였지만 Batch2에서는 40.81%로 커졌다. 오차를 셀별로 확인하니 39개 중 37개의 수명을 길게 예측했고, 500사이클 미만 28개는 평균 약 204사이클 과대 예측했다. 개발 데이터에는 500사이클 미만 셀이 하나도 없었다. **짧은 수명을 학습할 관측이 부족한 상태에서 그 영역까지 예측하게 한 점이 이번 설계의 한계였다.** 앞으로는 모델을 고르기 전에 학습 데이터가 예측 대상의 수명 범위와 입력 조건을 충분히 포함하는지 먼저 확인해야겠다고 생각했다.

입력 범위도 함께 살펴봤다. Batch2에서는 초기 용량 중앙값 21/39셀과 ΔQ log분산 11/39셀이 개발 데이터의 최댓값을 넘었다. 선정 모델의 용량 중앙값 계수는 양수여서 용량이 높으면 수명을 길게 예측하는 방향으로 작용한다. Batch2는 초기 용량이 높으면서 실제 수명은 짧은 셀이 많았기 때문에, 초기 용량의 절대값을 다른 Batch에도 같은 의미로 적용해도 되는지 의문이 생겼다. 분포 차이와 계수의 방향은 오차를 설명하는 근거지만, 이것만으로 용량 피처가 성능 저하의 주원인이라고 확정할 수는 없다.

피처를 추가하거나 제거한 결과도 배울 점이 있었다. 같은 ElasticNet·ln수명·alpha=0.01·l1_ratio=0.5에서 기본 A군은 6.755%, IQR을 제외한 D1군은 6.230%였다. 중복 피처를 점검한 선택이 이 CV 비교에서는 도움이 됐다. 반면 충전 조건을 추가한 C군은 같은 설정에서 7.280%였다. 최종 온도 계수는 0이 됐다. **EDA에서 관련이 있어 보이는 변수도 실제 예측에 얼마나 기여하는지는 별도로 확인해야 한다는 점을 배웠다.** 이 결과는 개발 표본 35개에서 얻은 비교이므로 다른 데이터에서도 같은 효과가 나타난다고 일반화하지 않는다.

이번 정책별 CV는 Batch1 안에서 새로운 충전 정책을 예측하는 상황을 평가했다. Batch2처럼 수명과 입력 분포가 다른 실험 집단으로 옮겨 가는 상황까지 충분히 검증하지 못했다. 따라서 CV가 낮다는 사실과 함께, 그 점수가 어떤 데이터와 분리 방식에서 나온 것인지 설명해야 한다. 논문의 9.1%와 비교할 때도 셀 구성·전처리·분리 조건을 확인해야 한다고 느꼈다. 이번에는 논문 성능에 도달하지 못했지만, 오차가 커지는 영역과 현재 설계가 다루지 못한 조건을 구체적으로 확인했다.

다음 실험에서는 아래 가설을 하나씩 검증해 보고 싶다. 아직 수행하지 않은 계획이며 성능 향상을 보장하는 결과는 아니다.

| 다음에 확인할 문제 | 실험해 볼 방법 | 확인할 결과 |
| --- | --- | --- |
| 학습에 단수명 영역이 부족함 | 단수명 셀과 다양한 충전 조건을 포함한 추가 학습 데이터를 확보하고, 평가 데이터는 별도로 고정 | 단수명 구간의 과대 예측·MAPE·MAE가 함께 줄어드는지 |
| 초기 용량의 절대값이 Batch 차이를 반영할 수 있음 | 기존 절대 용량 피처와 `Qd / Qd10` 같은 상대 용량 요약을 같은 개발 분할에서 비교 | Batch 차이에 덜 민감해지면서 수명 신호를 유지하는지. 상대화가 유용한 정보까지 없애는지도 확인 |
| 정책별 CV가 다른 Batch의 성능을 충분히 보여주지 못함 | 추가 데이터가 확보되면 Batch 단위로 학습·검증을 나누는 평가도 설계 | 같은 Batch 내부 점수와 다른 Batch에서의 점수 차이가 얼마나 큰지 |

이번 Batch2·3 평가 결과는 그대로 보존한다. 후속 실험에서 이미 확인한 평가 데이터로 설정을 다시 고르면 그 점수는 탐색 결과로 구분하고, 최종 성능은 별도의 미사용 데이터로 확인해야 한다. 다음 실험의 방향과 범위는 이 가설들을 비교한 뒤 정한다.

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
