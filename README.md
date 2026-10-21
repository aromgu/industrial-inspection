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

`--model`은 `patchcore`, `mhpatchcore`, `efficientad`, `dinomaly`, `dinomaly_l` 중에서 고릅니다.

## 베이스라인 실험

"AD 2에서 어떤 모델이 라인 투입 기준(미검 예산 대비 과검률)을 충족하는가"를 보기 위한 첫 비교입니다. 먼저 한 카테고리(`sheet_metal`)로 파이프라인을 검증하고, 이후 8개 카테고리로 확장합니다.

| 모델 | 계열 | anomalib 클래스 | 선정 이유 |
|---|---|---|---|
| EfficientAD | teacher-student 증류 | `EfficientAd` | AD 2 논문(IJCV 2026)의 표에서 평균 AU-PRO 58.7로 보고된 모델 |
| MH-PatchCore (Mahalanobis PatchCore) | 메모리뱅크 | `MHPatchcore` | PatchCore 계열 최신 변형. 공분산 화이트닝 + 스트리밍 coreset |
| Dinomaly (ViT-L) | DINOv2 재구성 | `Dinomaly(encoder_name="vit_large_patch14_reg4_dinov2")` | 원 논문에서 가장 강한 인코더 크기. AD 2 수치는 아직 보고된 곳을 찾지 못함 |
| PatchCore (기본) | 메모리뱅크 | `Patchcore` | 기준선 |

**학습 설정은 각 논문 값을 따릅니다.**

- EfficientAD: 70,000 iteration, 배치 1, Adam(lr 1e-4, wd 1e-5), 95% 지점(66,500)에서 lr 감소. 논문 부록 Algorithm 1과 anomalib 기본 스케줄이 일치합니다.
- Dinomaly: 카테고리별 개별 학습 5,000 iteration, 448 리사이즈 후 392 센터 크롭, StableAdamW(lr 2e-3 → 2e-4 cosine). 논문이 학습 배치 크기를 명시하지 않아서, ViT-L은 VRAM 16GB 때문에 배치 4로 실행합니다. **이 부분은 논문과 다를 수 있습니다.**
- MH-PatchCore, PatchCore: 학습 없이 특징 추출 한 번(1 epoch)으로 메모리뱅크를 만듭니다. anomalib 기본값입니다.

**평가 프로토콜**: 점수는 후처리 없이 저장하고(`score.py`), 임계값은 `test_public`을 층화 분할해 한쪽에서 정하고 다른 쪽에서 보고합니다(`evaluate.py`, 20회 반복). 미검 예산 0%, 1%, 5%일 때의 과검률과 AUROC를 비교합니다. 모든 실험은 seed 0, 단일 실행이므로 모델 간 작은 차이는 유의하지 않을 수 있습니다.

**결과** (sheet_metal, `test_public` 114장 = good 24 + defective 90, seed 0, 단일 실행)

과검률은 held-out 값입니다(`test_public`을 층화 분할해 한쪽에서 임계값을 정하고 다른 쪽에서 측정, 20회 평균). 미검 예산 0%와 1%가 같은 것은 보정 쪽 불량이 약 45장이라 1%로는 한 장도 놓칠 수 없기 때문입니다.

| 모델 | AUROC | 과검률 @ 미검 0% | 과검률 @ 미검 1% | 과검률 @ 미검 5% | 학습 시간 |
|---|---|---|---|---|---|
| Dinomaly (ViT-L) | **0.840** | 0.88 | 0.88 | 0.77 | 3,911 s * |
| PatchCore (기본) | 0.739 | 0.86 | 0.86 | 0.75 | 약 90 s |
| MH-PatchCore | 0.629 | 0.96 | 0.96 | 0.88 | 91 s |
| EfficientAD | 0.573 | 1.00 | 1.00 | 0.97 | 5,621 s * |

\* EfficientAD와 Dinomaly-L은 같은 GPU에서 동시에 학습했습니다. 서로 속도를 늦췄으므로 학습 시간끼리 비교할 수 없습니다(단독 실행 시 EfficientAD는 약 19 it/s, Dinomaly-L은 약 2.15 it/s).

![model comparison](docs/figures/compare__sheet_metal.png)

모델별 학습 곡선과 test 그래프는 [docs/figures/](docs/figures/)에 있습니다.

**읽는 법과 한계**

- **라인 투입 기준으로는 어느 모델도 부족합니다.** 가장 좋은 Dinomaly-L도 불량을 놓치지 않으려면 정상의 약 88%를 불량으로 판정해야 합니다. AUROC 순위와 과검률 순위는 거의 같지만, 격차의 크기는 AUROC만으로는 드러나지 않습니다.
- **EfficientAD는 학습할수록 test AUROC가 떨어졌습니다**(7,000 스텝 0.687 → 70,000 스텝 0.574, loss는 계속 감소). 원인은 확인하지 못했습니다. 가능한 가설은 AD 2의 조명 시프트가 섞인 test에 비해 정상 학습 분포가 좁아서 과적합되는 것이지만, 이 실험으로는 검증하지 않았습니다. 논문 설정(70,000 스텝)을 그대로 따른 결과입니다.
- **MH-PatchCore는 기본 PatchCore보다 낮았습니다.** 이 카테고리 한 곳의 단일 실행 결과이므로 일반화하지 마세요. 8개 카테고리 전체 스윕에서 다시 봐야 합니다.
- 학습 중 test AUROC 곡선은 그래프용이며 모델이나 임계값 선택에 쓰지 않았습니다.
- 표본이 작습니다(good 24장). 모델 간 작은 차이는 우연일 수 있습니다.

## 알려진 함정

- **경로에 비ASCII 문자가 있으면 anomalib이 거부합니다.** 경로 검증이 절대 경로로 바꾸기 전 문자열에 대해 이뤄지므로, 데이터 경로는 레포 기준 상대 경로로 넘기세요.
- **pandas 3과 anomalib 2.7은 호환되지 않습니다.** pandas 3에서 정상 샘플의 `mask_path`가 `None` 대신 `NaN`이 되어 학습 중 TypeError가 납니다. 그래서 `pandas<3`으로 고정했습니다.
- **점수는 후처리 없이 원시값으로 저장합니다.** anomalib 기본 후처리는 min-max 정규화 후 [0, 1]로 잘라내는데, 이러면 양 끝에 동점이 생겨 임계값 위치가 가려집니다.

## 로드맵

1. [x] 평가 모듈 (운영점, held-out 분할, 층화 부트스트랩)
2. [ ] 8개 카테고리 × EfficientAD / MH-PatchCore / Dinomaly 베이스라인 (sheet_metal 4종 검증 완료, 위 "베이스라인 실험" 참고. 이 카테고리 최고는 Dinomaly-L AUROC 0.84 — AD 2는 원래 SOTA도 AU-PRO 60% 미만인 어려운 벤치마크)
3. [ ] 소수 불량 보정에서 임계값이 과적합되는 문제 (마진, conformal 보정 비교)
4. [ ] 조명 시프트 강건성 (`test_public` 내 `shift_*`/`overexposed`/`underexposed` 조건별 분리 평가)
5. [ ] ONNX/TensorRT 변환, 지연시간과 정확도 트레이드오프
6. [ ] (선택) VLM 기반 불량 유형 설명
