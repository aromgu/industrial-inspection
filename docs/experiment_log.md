# 실험 로그 (로컬 머신)

이 컴퓨터(Ryzen 5 5600, RAM 32GB, RTX 3050 8GB)에서 돌린 실제 실험 기록. README의 "베이스라인 실험" 표는 다른 머신(RTX 5060 Ti 추정)에서 나온 결과이고, 이 문서는 이 머신에서 재현한 결과를 쌓아갑니다.

## 환경 세팅 변경 사항

- `pyproject.toml`: torch 인덱스를 `cu128` → `cu126`으로 변경. 이 머신의 드라이버(560.94)가 CUDA 12.6까지만 지원해서 cu128 빌드가 `torch.cuda.is_available()`을 거부함.
- Windows **Smart App Control**이 uv가 받아온 서명되지 않은 Python 바이너리 실행을 차단해서(`애플리케이션 제어 정책` 오류), 설정에서 꺼서 해결.
- `scripts/score.py`:
  - `--train-batch-size`, `--precision` CLI 옵션 추가 (작은 GPU에서 배치/정밀도 오버라이드용).
  - `Engine(..., enable_progress_bar=False, ...)` 추가. Lightning 기본 Rich 진행률 표시줄이 한글 Windows 콘솔(cp949)에서 유니코드 문자(•) 인코딩 실패로 크래시하는 문제 회피 (이미 자체 `StepProgress` tqdm 콜백이 있어서 기본 바는 불필요).
  - `torch.set_float32_matmul_precision("high")` 추가. RTX 3050(Ampere, Tensor Core)에서 fp32 matmul 속도 향상.

## 데이터셋

- `D:\datasets\MVTec_AD_2`에 공식 폴더 구조로 준비 (캐시도 `D:\datasets\.hf_cache`로 둬서 하드링크로 중복 저장 방지).
- 8개 카테고리, train 2,528 / validation 302 / test_public 1,084장 전부 확인됨.

## 하드웨어 기준 모델별 전략

| 모델 | 배치 | 비고 |
|---|---|---|
| PatchCore / MH-PatchCore | 16 (기본) | 학습 없음, 1 epoch 특징추출, 카테고리당 ~1.5~2분 |
| Dinomaly (ViT-B, 기본) | 8 (기본) | 8GB에서 OOM 없음 |
| Dinomaly-L (ViT-L) | 4 (기본, README는 16GB 가정) | smoke 테스트로 8GB에서 OOM 없이 확인됨 → 오버라이드 불필요 |
| EfficientAD | 1 (논문 고정) | 실측 ~9.5~10.2 it/s, 70,000 step 기준 카테고리당 약 2시간 |

## 실행 결과

| 모델 | 카테고리 | 학습 시간 | 비고 |
|---|---|---|---|
| EfficientAD | sheet_metal | 7,141s (~1h59m) | 실측, seed 0. `results/scores/efficientad__sheet_metal__seed0.csv` |

(이후 실행 결과는 계속 추가)
