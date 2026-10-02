# 원본 데이터 준비

[Kaggle 데이터셋](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle/data)에서 다음 3개 파일을 내려받아 프로젝트 루트의 `data-30/`에 둔다.

```text
data-30/
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
└── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

Batch1은 개발·학습, Batch2는 최종 평가, Batch3는 추가 평가에 사용한다. `2018-04-03_varcharge` 파일은 이번 실험 대상이 아니다. 용량이 큰 원본 파일은 저장소에 포함하지 않는다. MATLAB v7.3(HDF5) 자료에서 `h5py`로 초기 사이클 1–100을 읽는다.

원본 139셀 중 총수명이 확인된 129셀을 학습·평가에 사용한다. Batch2 8셀·Batch3 2셀의 수명 결측은 임의 대치하지 않는다. Batch1의 비어 있는 첫 기록은 결측으로 표시하고 원래 사이클 번호는 유지한다. 스파이크·장수명 셀은 유지한다. 제공된 `cycle_life`가 목표값이며 마지막 관측 사이클로 대체하지 않는다.
