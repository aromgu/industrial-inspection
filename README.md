# industrial-inspection

MVTec AD 2 기반 비지도 외관검사. "AUROC 몇 점"이 아니라 **라인에 투입할 수 있는가**를 묻습니다.

- **운영점 평가**: 미검률 예산(0%, 1%, 5%)을 고정했을 때의 과검률
- **Held-out 임계값**: `test_public`을 층화 분할해서, 한쪽에서 임계값을 정하고 다른 쪽에서 보고합니다. 분할을 반복해 분포(평균, p95)로 제시하고, 같은 세트로 보정했을 때 생기는 낙관 편향도 함께 기록합니다.
- **강건성**: 조명이나 위치가 바뀔 때의 성능 변화 (`test_public`에 섞여 있는 `regular`/`overexposed`/`underexposed`/`shift_1~4` 조건으로 분석)
- **추론 지연**: ONNX/TensorRT로 변환해 장당 지연시간 측정

## 환경

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- PyTorch cu128 (RTX 50 시리즈 / Blackwell sm_120은 CUDA 12.8 이상 필요)

```
uv sync
uv run pytest
```

## 데이터

MVTec AD 2 원본: [공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), CC BY-NC-SA 4.0. **2026-10 기준 MVTec의 자체 다운로드 서버(mydrive.ch)가 전체 다운로드·카테고리별 다운로드 모두 404를 반환**합니다(anomalib에 내장된 링크도 동일 증상). MVTec에 문의는 넣어뒀지만, 그동안은 비공식 미러를 씁니다.

```
uv run python scripts/prepare_ad2_mirror.py --root datasets/MVTec_AD_2
```

- 소스: [huggingface.co/datasets/pjramg/MVTecAD2_FO](https://huggingface.co/datasets/pjramg/MVTecAD2_FO) (FiftyOne export, 비공식 미러)
- `train`(2,528) / `validation`(302) / `test_public`(1,084 = good 379 + bad 705, 픽셀 단위 GT 마스크 포함) 8개 카테고리 전부 포함. **`test_private`/`test_private_mixed`는 원래도 공개된 적 없는 데이터라 어느 미러에도 없음** — 로컬 평가는 `test_public`만으로 충분합니다.
- GT 마스크는 `samples.json`에 zlib 압축된 numpy 배열(FiftyOne Segmentation 필드)로 들어 있어, 스크립트가 이를 디코딩해 anomalib이 기대하는 `ground_truth/bad/*_mask.png`로 풀어줍니다.
- 전체 약 16GB. 공식 링크가 복구되면 이 스크립트 없이 공식 아카이브를 그대로 풀어도 코드는 동일하게 동작합니다.

## 실행

```
uv run python scripts/score.py --model patchcore --category sheet_metal --root datasets/MVTec_AD_2
uv run python scripts/evaluate.py results/scores/*.csv
```

## 알려진 함정

- **경로에 비ASCII 문자가 있으면 anomalib이 거부합니다.** 경로 검증이 절대 경로로 바꾸기 전 문자열에 대해 이뤄지므로, 데이터 경로는 레포 기준 상대 경로로 넘기세요.
- **pandas 3과 anomalib 2.7은 호환되지 않습니다.** pandas 3에서 정상 샘플의 `mask_path`가 `None` 대신 `NaN`이 되어 학습 중 TypeError가 납니다. 그래서 `pandas<3`으로 고정했습니다.
- **점수는 후처리 없이 원시값으로 저장합니다.** anomalib 기본 후처리는 min-max 정규화 후 [0, 1]로 잘라내는데, 이러면 양 끝에 동점이 생겨 임계값 위치가 가려집니다.

## 로드맵

1. [x] 평가 모듈 (운영점, held-out 분할, 층화 부트스트랩)
2. [ ] 8개 카테고리 × PatchCore / EfficientAD / Dinomaly 베이스라인 (sheet_metal/PatchCore 기본값: AUROC 0.74 — AD 2는 원래 SOTA도 AU-PRO 60% 미만인 어려운 벤치마크)
3. [ ] 소수 불량 보정에서 임계값이 과적합되는 문제 (마진, conformal 보정 비교)
4. [ ] 조명 시프트 강건성 (`test_public` 내 `shift_*`/`overexposed`/`underexposed` 조건별 분리 평가)
5. [ ] ONNX/TensorRT 변환, 지연시간과 정확도 트레이드오프
6. [ ] (선택) VLM 기반 불량 유형 설명
