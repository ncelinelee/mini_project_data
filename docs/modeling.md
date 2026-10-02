# 모델과 하이퍼파라미터 설명

## 문제 정의와 입력

배터리 한 개를 한 행으로 만들고 원래 사이클 번호 1-100에서 얻은 정보로 총수명 `cycle_life`를 예측한다. 단위는 사이클이다. 같은 셀의 사이클을 서로 다른 학습·검증 행으로 나누지 않는다.

Batch1의 550사이클 기준 분류는 단수명 1개·장수명 45개로 불균형하므로 연속 수명을 예측하는 회귀를 선택했다. 원수명과 자연로그 수명 `ln(cycle_life)`를 비교한다. 로그 모델의 예측에는 `exp`를 적용한 뒤 모든 지표를 원래 사이클 단위에서 계산한다. 별도의 역변환 보정이나 예측값 잘라내기는 적용하지 않는다.

## 피처를 선택한 이유

| 피처 | 계산 | 선정 이유와 주의점 |
|---|---|---|
| `deltaq_log10var` | 동일한 1,000점 전압 격자에서 Q100(V)-Q10(V)의 표본분산(ddof=1)에 log10 | 초기 곡선 변화와 수명의 강한 음의 관계. 원분산의 작은 수치 범위를 로그로 표현 |
| `deltaq_min` | ΔQ 곡선의 최솟값(Ah) | log분산과 중복되므로 B군에서 대체하여 비교 |
| `qd_median` | 유효한 초기 방전 용량의 중앙값(Ah) | 초기 용량 수준. 스파이크에 평균보다 덜 민감 |
| `qd_iqr` | 초기 Qd의 Q75-Q25, 선형 보간(Ah) | 초기 변동 폭. 용량 변화량과 중복될 수 있음 |
| `qd_change100_10` | 원래 Qd100-Qd10(Ah) | 초기 변화 방향과 크기. 두 시점의 측정에 민감 |
| `temp_mean` | 초기 Tavg의 평균(°C) | 초기 온도 신호. 최고온도와 중복되어 둘을 동시에 기본군에 넣지 않음 |
| `charge_median` | 초기 충전시간 중앙값(분) | 충전 조건 추가가 도움이 되는지 C군에서 검증 |
| `policy_c1`, `policy_c2`, `policy_switch_pct` | 정책 문자열의 1·2단계 C-rate와 전환 SOC(%) | 정책의 수치적 조건. 수명과의 관계가 Batch마다 달라 별도 비교 |

A는 log분산·용량 중앙값·IQR·변화량·평균온도 5개다. B는 log분산을 최솟값으로 대체한다. C는 A에 충전시간과 정책 변수 4개를 추가한다. D1은 A에서 IQR, D2는 A에서 변화량을 제외한다.

피처 간 통계적 독립성을 확보했다고 주장하지 않는다. IQR과 변화량의 Pearson r은 Batch1 -0.961, Batch2 -0.408, Batch3 -0.993이다. 중복을 점검하고, 규제와 피처 제거 비교로 예측 영향을 검증한다. IR은 추가 0값의 의미가 불명확해 입력에서 보류한다. 전체 수명 곡선의 knee와 관측 종료 시점은 모델 입력에 사용하지 않는다.

## 모델 후보와 EDA의 연결

| 모델 | 비교 이유 | 학습하는 값 |
|---|---|---|
| Mean | 피처 없이 예측하는 기준보다 개선되는지 확인 | 원수명 평균 또는 ln수명의 평균(역변환하면 기하평균) |
| Univariate | 수명과 강한 관계를 보인 ΔQ log분산 한 개의 설명력 확인 | 계수와 절편 |
| Ridge | 개발 표본 35개와 중복 피처에서 계수 변동을 줄임 | L2 규제가 적용된 계수와 절편 |
| ElasticNet | 중복 피처에서 계수 축소와 일부 계수 0화를 함께 비교 | L1·L2 규제가 적용된 계수와 절편 |
| Random Forest | 비선형 관계와 피처 조합이 선형 모델보다 유리한지 비교 | 부트스트랩으로 학습한 여러 트리의 분기 |
| Gradient Boosting | 앞선 모델의 잔차를 보완하는 비선형 예측 비교 | 순서대로 추가하는 트리의 분기 |

정형 데이터이며 표본이 적어 딥러닝은 사용하지 않는다. 트리 모델은 학습 타깃 범위를 넘는 장수명 외삽이 제한될 수 있으므로 Batch3의 장수명 오차를 별도로 해석한다. 모델 선정은 이러한 예상만으로 정하지 않고 공통 교차검증으로 결정한다.

## Grid Search의 조정 항목

| 모델 | 파라미터 | 비교값 | 역할 |
|---|---|---|---|
| Ridge | `alpha` | 0.01, 0.1, 1, 10, 100 | L2 규제 강도. 커질수록 계수를 줄이지만 너무 크면 과소적합 가능 |
| ElasticNet | `alpha` (원수명) | 0.1, 1, 10, 100 | L1·L2 전체 규제 강도 |
| ElasticNet | `alpha` (ln수명) | 0.0001, 0.001, 0.01, 0.1 | 목표값의 단위와 규모 차이를 고려한 규제 범위 |
| ElasticNet | `l1_ratio` | 0.2, 0.5, 0.8 | L1의 비중. 커지면 일부 계수를 0으로 만드는 성향이 커짐 |
| Random Forest | `max_depth` | 2, 4 | 트리 최대 깊이. 깊을수록 복잡한 관계를 표현 |
| Random Forest | `min_samples_leaf` | 3, 6 | 말단 노드 최소 셀 수. 커질수록 작은 집단에 맞춘 예측을 제한 |
| Gradient Boosting | `learning_rate` | 0.03, 0.1 | 트리 하나가 예측을 수정하는 크기 |
| Gradient Boosting | `n_estimators` | 100, 200 | 순서대로 추가하는 트리 수. 학습률과 함께 비교 |
| Gradient Boosting | `max_depth` | 1, 2 | 각 트리의 복잡도. 1은 단순 분기, 2는 제한적인 상호작용 |

값의 범위는 설계 단계에서 고정한 탐색 범위이며, 범위 밖까지 포함하는 절대적인 최적값을 의미하지 않는다. 한 피처군·목표 변환에서 29개 조합, 5개 피처군 × 2개 목표 변환 × 5-fold = 1,450회 튜닝 학습을 한다. 두 기준 모델 × 두 변환 × 5-fold의 20회가 더해져 CV 학습은 총 1,470회다.

## 고정 설정과 이유

| 적용 | 고정 설정 | 역할과 이유 |
|---|---|---|
| 선형 모델 | `StandardScaler`, `fit_intercept=True` | 피처 단위가 규제 크기를 좌우하지 않도록 학습 폴드의 평균·표준편차로 표준화. 절편 학습 |
| ElasticNet | `max_iter=10000`, `tol=1e-4`, `selection='cyclic'` | 반복 상한·수렴 허용오차·계수 갱신 순서. 경고와 실제 반복 수 기록 |
| Random Forest | `n_estimators=300`, `bootstrap=True` | 중복 허용한 셀 표본으로 여러 트리를 만들고 평균화 |
| Random Forest | `max_features=1.0`, `criterion='squared_error'` | 모든 입력 피처를 분기 후보로 보고 제곱오차 감소로 분기 |
| Gradient Boosting | `min_samples_leaf=3`, `loss='squared_error'` | 아주 작은 말단 노드를 제한하고 잔차의 제곱오차를 줄임 |
| Gradient Boosting | `subsample=1.0`, `n_iter_no_change=None` | 각 단계의 학습 셀 전체 사용. 추가 내부 검증셋 대신 공통 CV로 트리 수 비교 |
| 분리·트리 모델 | `random_state=42` | 같은 실험 재현. 좋은 seed를 탐색하지 않음 |
| 교차검증 | `GroupKFold(n_splits=5, shuffle=False)` | 원문 충전 정책 문자열로 분리. 같은 정책이 한 폴드의 학습·검증에 겹치지 않음 |

모델의 학습 목적함수는 제곱오차 기반이며, 설정을 선택하는 검증 지표는 MAPE다. 둘을 구분한다. 모든 API 기본값은 실행 버전과 함께 `day2/results/estimator_parameters.json`에 저장한다.

## 선정과 평가 절차

1. Batch1 46셀에서 정책 그룹의 20%를 별도 검증으로 고정한다(seed=42). 개발 35셀·18정책, Valid 11셀·5정책이다.
2. 개발 데이터에만 정책별 5-fold CV를 적용한다. 모든 후보가 같은 분할을 사용한다.
3. 각 폴드 학습 데이터에만 표준화와 모델을 fit한다. 로그 예측은 역변환한다.
4. 각 폴드의 원래 단위 MAPE(%)를 계산하고, 5개 값의 산술평균을 선정 점수로 사용한다. 폴드 크기와 무관하게 각 가중치는 1/5이다.
5. 반올림 전 평균 MAPE 최솟값으로 모델·피처군·변환·파라미터를 함께 선택한다. 전체 OOF MAPE, MAE, RMSE, 폴드 표준편차는 보조 지표다.
6. 선택 결과를 파일에 고정한 뒤 개발 35셀 전체로 한 번 학습한다. 같은 모델로 Fit(학습 셀 재예측) 35셀, Valid 11셀, Batch2 39셀, Batch3 44셀을 평가한다. 평가 결과로 재튜닝하지 않는다.
7. Valid permutation importance는 입력을 섞었을 때 MAPE가 얼마나 증가하는지 30회 반복한다. 해석에만 사용하며 모델을 바꾸지 않는다. 중복 피처에서는 중요도가 분산될 수 있으며 인과효과가 아니다.

제출 표의 Train (Batch 1 CV)은 개발 35셀에 대한 정책별 5-fold MAPE의 산술평균이다. Fit_Batch1은 같은 35셀을 다시 예측한 오차로 별도의 보조 지표이며 Train에 대입하지 않는다. Valid는 개발과 충전 정책이 겹치지 않는 11셀 hold-out이다. 일반적인 셀 무작위 CV와 달리 이번 GroupKFold도 충전 정책을 분리한다. MAPE는 상대 오차, MAE는 사이클 단위의 오차, RMSE는 큰 오차에 더 민감한 지표다. 오차 차이는 퍼센트포인트(pp)로 보고한다.

## 성능 표와 Gap 해석

필수 표는 `day2/results/model_performance.csv`, Batch3 추가 표는 `model_performance_batch3.csv`다. 과제의 행 이름을 유지하고, MAPE가 커질수록 나빠지는 점에 맞춰 Gap 계산식을 명시한다.

- Gap (Train-Valid) = Valid − Train(CV)
- Gap (Valid-Test) = Test(Batch2) − Valid
- Gap (Target-Test) = 해당 Test − 9.1%
- Gap (Batch2-Batch3) = Test(Batch2) − Test(Batch3)

모든 Gap은 퍼센트포인트(pp)다. 내부 검증 Gap의 양수는 검증 오차 악화를 뜻하지만, Train이 CV 오차이므로 학습 오차와 검증 오차의 전형적인 과적합 판단과 동일하지 않다. 후보 선정의 낙관성, 소표본 변동, 정책 구성 차이가 함께 작용할 수 있다. Batch2−Batch3의 양수는 Batch2 오차가 더 크다는 뜻이다. 서로 다른 수명 분포와 MAPE의 분모를 고려해 해석한다.

## Batch3의 추가 검증 한계

ΔQ는 같은 셀의 cycle100과 cycle10을 공통 전압 격자에서 뺀다. 두 사이클에 동일한 상수 용량 오프셋이 있다면 차분에서 상쇄되지만, 전압 격자의 일치만으로 물리적 곡선 시작점이 정렬됐다고 볼 수 없다. 배치별 시작점 차이와 사이클에 따라 달라지는 오프셋의 보정은 검증하지 않았다. 후속 실험에서는 원시 곡선의 시작 조건을 먼저 확인하고 일관된 정렬 기준을 검증해야 한다.

수명 미확인 셀과 비어 있는 첫 기록은 지정한 기준으로 처리했다. 스파이크·장수명 셀은 오류라는 근거가 없어 보존하고 구간별 오차로 영향을 확인했다. 저자 코드의 연속 실험 연결·품질 제외를 그대로 재현하지 않았으므로 원논문과 동일한 정제 데이터에 대한 성능 비교는 아니다.

## 참고

- [Severson et al. (2019), 논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)
- [저자의 공개 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [TransformedTargetRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
