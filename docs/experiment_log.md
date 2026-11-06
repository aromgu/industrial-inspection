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

### 학습 시간

| 모델 | 카테고리 | 학습 시간 | 비고 |
|---|---|---|---|
| EfficientAD | sheet_metal | 7,141s (~1h59m) | 실측, seed 0 |
| PatchCore / MH-PatchCore | 8개 카테고리 전부 | 카테고리당 ~4~5분 (2모델 합산) | README 추정(~90s)보다 느림. RTX 3050 특징추출 속도 차이로 추정 |

### AUROC / 과검률 (미검 예산 0%, held-out, 20회 평균) — 8개 카테고리 전체

| 모델 | 카테고리 | AUROC | 과검률 | 비고 |
|---|---|---|---|---|
| PatchCore | can | 0.450 | 0.981 | |
| PatchCore | fabric | 0.714 | 0.827 | |
| PatchCore | fruit_jelly | 0.769 | 0.800 | |
| PatchCore | rice | 0.552 | 0.921 | |
| PatchCore | sheet_metal | 0.712 | 0.892 | README 원 실험(0.739)과 근접 — 파이프라인 정상 동작 확인 |
| PatchCore | vial | 0.875 | 0.494 | |
| PatchCore | wallplugs | 0.491 | 0.993 | |
| PatchCore | walnuts | 0.849 | 0.920 | |
| MH-PatchCore | can | 0.478 | 0.993 | |
| MH-PatchCore | fabric | 0.443 | 0.992 | |
| MH-PatchCore | fruit_jelly | 0.913 | 0.600 | |
| MH-PatchCore | rice | 0.537 | 0.933 | |
| MH-PatchCore | sheet_metal | 0.650 | 0.971 | README 원 실험(0.629)과 근접 |
| MH-PatchCore | vial | 0.908 | 0.168 | |
| MH-PatchCore | wallplugs | 0.526 | 0.998 | |
| MH-PatchCore | walnuts | 0.864 | 0.813 | |

전체적으로 `can`/`fabric`/`rice`/`wallplugs`는 두 모델 다 AUROC 0.5 근처(=거의 우연 수준)라 라인 투입 기준에 한참 못 미침. `vial`/`fruit_jelly`/`walnuts`는 AUROC 0.85~0.91로 상대적으로 양호. 8개 카테고리 간 난이도 차이가 sheet_metal 한 곳만 보던 README 원 실험보다 훨씬 크게 드러남.

### Dinomaly (ViT-B, 기본) — 진행 중 중단 시점 (4/8)

| 카테고리 | AUROC | 과검률 |
|---|---|---|
| can | 0.515 | 0.996 |
| fabric | 0.682 | 0.750 |
| fruit_jelly | 0.817 | 0.745 |
| rice | 0.683 | 0.933 |

카테고리당 학습 시간 **~55~65분** (1.3~1.4 it/s, 5,000 step) — README가 암시한 속도보다 훨씬 느림. 8개 전체면 ~8시간 예상.

### EfficientAD

sheet_metal AUROC 0.598 — README 원 실험(0.573, RTX 5060 Ti)과 비슷한 범위. "학습할수록 test AUROC가 떨어진다"는 README의 관찰과 일치하는 방향.

## 2026-10-28 작업 중단 시점 요약

사용자 지시로 학습을 중단하고 지금까지 결과를 정리함. 재개 방법은 `scripts/sweep.sh`의 사용법 주석과 README 참고.

## 2026-10-30 RTX 5060 Ti 머신에서 이어서 진행

- 환경: RTX 5060 Ti 16GB(Blackwell). 이 GPU는 cu128 빌드가 필요해서, 레포가 cu126으로 고정된 상태에서는 `uv sync`를 하지 않고 venv의 Python을 직접 호출함(`PY=.venv/Scripts/python.exe`). `sweep.sh`의 `uv run`은 이 머신 PATH에 uv가 없어 모든 조합이 exit 127로 즉시 실패했었음 → `PY` 변수로 우회.
- 사용자가 같은 PC에서 게임을 해서 `--game-mode`(GPU 듀티 약 60%, VRAM 60% 상한, 워커 2개, 낮은 CPU 우선순위)로 실행. Dinomaly-L 실측 평균 GPU 사용률 약 60%, 카테고리당 약 49~53분.
- 다른 머신의 점수 CSV는 git에 없어서 PatchCore/MH-PatchCore 8개 카테고리를 이 머신에서 다시 돌림. sheet_metal 기준 머신 간 AUROC 차이는 최대 약 0.03(PatchCore 0.712 vs 0.739, MH-PatchCore 0.650 vs 0.629).

### Dinomaly-L (ViT-L) 8개 카테고리

| 카테고리 | AUROC | 과검률 @ 미검 0% | 학습 시간 |
|---|---|---|---|
| can | 0.503 | 0.99 | 2,999s |
| fabric | 0.698 | 0.86 | 3,180s |
| fruit_jelly | 0.870 | 0.91 | 2,940s |
| rice | 0.769 | 0.83 | 3,175s |
| sheet_metal | 0.840 | 0.88 | 3,911s (EfficientAD와 동시 학습) |
| vial | 0.810 | 0.64 | 2,962s |
| wallplugs | 0.457 | 0.92 | 3,030s |
| walnuts | 0.715 | 0.71 | 3,181s |

세 모델 비교, 조건별 분석, 해석상 주의점은 README의 "8개 카테고리 결과" 참고. 데이터 특성(결함 크기, 조건별 밝기, 부품 수)은 `scripts/data_profile.py`와 `docs/figures/data_profile.png`.

## 2026-11-02 추가 실험

- `evaluate.py`를 부품 단위 분할로 변경. 미검 0% 임계값의 처음 보는 부품 미검률 평균 5.3%(p95 17%), 이미지 단위일 때는 2.1%(p95 7%).
- 소수 불량 임계값 보정, 조건별 임계값: 결과는 README "추가 실험" 참고.
- 타일 MH-PatchCore: can 0.502→0.504, wallplugs 0.537→0.538. 결함 크기 가설은 직접 실험에서 확인되지 않음.
- 조명 증강 PatchCore(밝기·대비 ±35%): 8개 카테고리 평균 AUROC 0.679→0.673.
- 지연시간(배치 1): EfficientAD 7.4/6.0ms, MH-PatchCore 14.7/9.8ms(ONNX 점수 불일치), PatchCore 14.1/13.7ms, Dinomaly-L 90.5/52.4ms (PyTorch/ONNX Runtime).
- EfficientAD 8개 카테고리 스윕을 게임 모드로 시작(can 완료: AUROC 0.494, 5,880s). fabric이 67%(47075/70000)일 때 2026-11-02 18:17에 사용자 요청으로 중단. 체크포인트가 없어 fabric은 처음부터 다시 돌려야 함. 남은 6개(fabric, fruit_jelly, rice, vial, wallplugs, walnuts)는 미실행. 카테고리당 약 100분(게임 모드), 재개 명령은 `scripts/sweep.sh` 주석 참고.

## 2026-11-03

- EfficientAD 나머지 6개 카테고리를 게임 모드 없이 두 줄 병렬로 완료(합계 23.7 it/s, 한 줄 약 19 it/s). 8개 결과: can 0.494, fabric 0.547, wallplugs 0.534, rice 0.504, sheet_metal 0.573, fruit_jelly 0.873, vial 0.804, walnuts 0.760. 평균 0.636.
- 조명 증강 PatchCore의 shift 조건 과검률 0.909 → 0.908(효과 없음).
- 원본 MVTec AD(공식 아카이브와 SHA-256 일치)로 PatchCore 15개 카테고리: 평균 이미지 AUROC 0.982. 파이프라인 정상.
- 이상 맵 분석(4×8): 재현 26/32 동일, 나머지 차이 ≤0.0006. 집계 방식에 일관된 개선 없음. AU-PRO@0.05(모델 입력 해상도) 평균 Dinomaly-L 0.317, MH-PatchCore 0.211, PatchCore 0.176, EfficientAD 0.141.
- MH-PatchCore 원본 MVTec AD 15개: 평균 0.986(PatchCore 0.982). 구현 정상.
- PatchCore 392px(can, fabric, rice, sheet_metal): AUROC 평균 0.610 → 0.627, AU-PRO 0.059 → 0.150. rice·sheet_metal에서 Dinomaly-L 우위는 392px와 비교해도 남음 → 우위는 주로 모델에서 옴.
- validation 정상 이미지로 정한 conformal 임계값(과검 10% 목표): PatchCore 과검 20%/미검 53%, Dinomaly-L 44%/28%. regular 조명만 보면 7%, 11%로 목표 근처, shift 조명에서 31%, 66%.
- 부품 단위 부트스트랩(300회): 모델 평균 AUROC 구간이 겹침. Dinomaly-L 0.71 [0.67, 0.74], EfficientAD 0.64 [0.61, 0.66].
- 이미지 읽기+리사이즈(CPU): 이미지당 43~194ms.
- 타일 재실험(뱅크 9000)은 MH-PatchCore 제외로 취소. seed 반복도 하지 않음.
- 연구 단계 시작: 모델 PatchCore, Dinomaly-L. Q2(조명 과검, walnuts → vial, sheet_metal), Q1(작은 결함, can → wallplugs). 부품 단위 개발/평가 절반 분할(seed 2026).
- Q2-1 맵 자기 정규화(walnuts 개발용 절반): 기각. PatchCore shift 과검 1.00 → 최선 0.87, Dinomaly-L 변동 2.36 → (중앙값 빼면) 3.29. 조명 효과는 국소적.
- Q2 H2-b/H2-c/H2-d 코드 작성(실행은 2026-11-04 예정): `lighting.py`(LightNorm 5종, LocalLight), `score.py` 변형 이름(`<base>_<norm>`, `<base>_light`, `<base>_light_<norm>`), `q2_compare.py`, `q2_preview.py`, `q2_run.sh`. 스모크 테스트 통과(patchcore_light_clahe, dinomaly_l_light_retinex, walnuts). 테스트 16개 통과.

## 2026-11-05

- Q2 H2-b(PatchCore 정규화 5종, walnuts 개발용 절반, 게임 모드, 변형당 약 5분): 주 지표(미검 0% 임계값의 shift 과검)는 보통 조명에서도 1.00이라 판단 불가. 조명 오경보 지표(`fa_shift`, 보통 조명 정상 최고점 기준) 추가. 기준 0.40 → 흑백 0.13(차이 −0.27 [−0.53, 0.00], 놓침 10% 그대로), 표준화 0.00(차이 −0.40 [−0.80, −0.10], 대신 놓침 38%). CLAHE·평활화·Retinex는 효과 없음. 확실한 승자 없음, 흑백이 후보.
- 경고: 정규화가 들어간 pre-processor(Compose)는 체크포인트에 저장되지 않음(weights_only). 체크포인트로 이상 맵을 다시 뽑을 때는 `score.MODELS`로 모델을 만들어야 함.
- Q2 H2-c(국소 조명 증강, PatchCore, walnuts 개발용 절반, 변형당 약 5분): 증강만은 오경보 0.40 → 0.47(효과 없음). 증강+흑백은 0.07(차이 −0.33 [−0.67, −0.07])이지만 놓침 10% → 24%. 흑백 단독(0.13, 놓침 10%)이 여전히 최선 후보.
- H2-d 시작: dinomaly_l_gray, dinomaly_l_light (게임 모드).
- Q2 H2-d(Dinomaly-L, walnuts 개발용 절반): dinomaly_l_gray는 게임 모드(1.65 it/s, 53분), 사용자가 GPU 전체 사용을 허락해 큐 셸만 멈추고 dinomaly_l_light는 전체 GPU(2.6~2.85 it/s, 34분). 조명 오경보 기준 0.73 → 흑백 0.80, 증강 0.73(변동 2.55 → 2.14배). 둘 다 효과 없음. 성공한 방법이 없어 평가용 절반·vial·sheet_metal 확인은 하지 않음.
- Q2 H2-e(새 조명에서 정상품 k개 재촬영, walnuts 개발용 절반, 재학습 없음): PatchCore bank k=1 오경보 0.40 → 0.28(놓침 9.5 → 12%), thr k=2 0.28(11%). Dinomaly-L thr는 오경보 0.73 → 0.40/0.29지만 놓침 43/57%. PatchCore 재채점 점수는 저장 점수와 상대 오차 0.15%, 순위 상관 0.9997(이미지 읽기 경로 차이).
- Q1 H1-a 시작: patchcore_{256,392,512}_c01(코어셋 1%, PatchCore 논문 설정) can. `export_maps.py --eval-side 512`: 모든 해상도의 맵을 512 격자로 맞추고 마스크는 원본 해상도 파일에서 max-pool로 만들어 작은 결함이 사라지지 않게 함. 정상만 든 배치에서 `mask_path`가 None이라 실패하던 것 수정(스모크는 앞쪽 배치만 돌려 못 잡음).
- Q1 결과(can): 256/392/512(코어셋 1%) AUROC 0.47/0.51/0.53, AU-PRO(512 격자, 원본 마스크) 0.010/0.058/0.091. 512는 패치 170만 개라 대표 패치 선택 62분, GPU 메모리 15.7GB. 크기별 AUROC 모두 구간이 0.5 포함. 위치별 정규화(validation 정상 46장) 효과 없음. 맵을 직접 보면 결함은 보이지만 홀로그램 라벨 영역이 더 밝음. 보통 조명에서도 AUROC 0.44, test 정상 캔 점수 > validation 정상 캔. 결론: 원인은 결함 크기가 아니라 캔마다 다른 정상 변화. 타일(H1-b)은 하지 않음.
- Q2 H2-f(Dinomaly-L 미세조정, 새 조명 정상 1개 + 학습 사진 반반, 200스텝, StableAdamW lr 2e-4, 회당 약 95초 × 15회): 조명 오경보 0.73 → 0.53, 놓침 14 → 18%. shift_1 1.00 → 0.60, shift_3 1.00 → 0.70, shift_2 변화 없음. 재채점 점수는 저장 점수와 상대 오차 3e-4.
- Q2 최종 확인(H2-e, 재학습 없음): walnuts 평가용 절반에서 PatchCore 메모리 뱅크 추가 0.60 → 0.58(재현 안 됨), vial 0.10 → 0.16, sheet_metal 0.83 → 0.86. 임계값 다시 맞추기는 오경보를 줄이지만 놓침이 늘어남(walnuts PatchCore 4 → 8%, sheet_metal 2 → 17%). 개발용 절반 승자는 우연이었음. walnuts 평가용 절반은 이 방법으로 한 번 사용함.
- Q2 H2-f 다른 카테고리 확인(Dinomaly-L 미세조정, 전체 부품): vial 0.65 → 0.63(20회), sheet_metal 1.00 → 1.00(12회). 재현 안 됨. Q2 종료: 가벼운 대응으로는 조명 과검이 줄지 않음.
- Q1 wallplugs(게임 모드, PatchCore 256 체크포인트, 맵만 다시 뽑음): test 정상 = validation 정상 점수(46.0 vs 46.3), 보통 조명 AUROC 0.61. 부품 자세가 사진마다 다르고 큰 파손 결함도 맵에서 어두움. `export_maps.py --game-mode` 추가.
- 새 연구, 촬영 조건 간 일관성(`xcond.py`, 재학습 없음, 기존 맵, 부품 단위, 부트스트랩 500): Dinomaly-L AUROC 0.771(사진 1장) → 0.777(점수 평균) → 0.819(pixel-min), 사진 2장 0.800. 다른 세 모델은 pixel-min이 점수 평균보다 낫지 않음. 구간이 0을 벗어난 카테고리: Dinomaly-L can +0.36, wallplugs −0.06. 엄격한 운영점(정상 경보 0)에서 잡는 불량 부품 70 → 69.
- 조건 간 일관성 보완: 사진 간 이동량(경계 위상 상관) 256 격자 기준 대부분 0칸, wallplugs 최대 0.28칸, vial 최대 0.5칸 → 위치 맞추기 불필요. 결함 영역 점수는 조건 간 거의 일정. anchored 0.773, hybrid 0.798(엄격 기준 72개), pixel-min 0.819로 pixel-min 유지.
- 결함 영역 F1 평가(`seg_eval.py`, 공개 test, SuperADD 방식: 불량 사진만, 1/4 해상도): Dinomaly-L 23.7(공식 임계값)/34.1(최적 임계값 상한), MH-PatchCore 19.5, PatchCore 18.6, EfficientAD 16.6. SuperADD 보고값 62.6. 정상 포함 시 Dinomaly-L 21.0. validation 맵 32개를 게임 모드로 추가 export(`export_maps.py --split validation`이 이제 항상 맵 저장).
- HiResKNN(DINOv2-L, 짧은 변 672, 448 조각 112 겹침, 4개 층, 참고 사진 16장, 게임 모드 약 30분): 결함 영역 F1 28.1(공식 임계값)/32.3(SuperADD 규칙)/40.3(최적 임계값 상한), AU-ROC@0.05 77.7. Dinomaly-L 23.7보다 좋지만 SuperADD 62.6과 차이 큼. rice는 공식 24.7 vs 최적 69.2로 임계값 영향이 큼.
- 공식 MVTec AD 2 데이터로 교체: 미러와 4,619장 모두 픽셀 동일, 파일 이름 접미사(-NN)만 다름. 결과 파일 경로 402개 수정. 비공개 test 두 세트 추가.
- 임계값·후처리 8개 카테고리 공통 설정(seg_post.py, hires_dinov2l): p95x1.2+fill+small256으로 평균 F1 28.1 → 40.6. fabric 28→73, rice 25→58 크게 개선, wallplugs는 6.0→7.7로 거의 그대로(전체 공통 배율이 wallplugs에는 부족).
- wallplugs 단독 포커스: (1) wallplugs만 맞춘 임계값·후처리(p95x1.6+close6+fill+small64) F1 6.0→22.7, 모델 변경 없이 약 4배. (2) 회전 메모리 뱅크(16 refs x 8 rotations=128 ref-equivalents, HiResKNN.fit rotations=): 가설(플러그가 모든 각도라 뱅크가 각도를 못 봄)이 틀림 — 사진 한 장에 이미 모든 각도의 플러그가 있어 뱅크 다양성이 이미 충분했음. 결과는 전부 소폭 악화(F1 공식 6.0→4.0, 최적 24.0→23.0, AU-ROC 0.804→0.796). HiResKNN.score에서 q/bank 모두 청킹하도록 고쳐 OOM 해결(원래 cdist가 (q_chunk, 전체 bank) 행렬을 한 번에 만들어 8배 커진 뱅크에서 3.57GiB 할당 실패).
- wallplugs 참고 사진 48장(회전 없음, 서로 가장 다른 48장): F1 공식 4.8/최적 22.3/AU-ROC 0.795, 전부 16장 기준(6.0/24.0/0.804)보다 소폭 하락. 전용 임계값·후처리 최선 23.6으로 16장 때(22.7)와 비슷. 회전과 마찬가지로 뱅크 다양성·크기는 wallplugs 병목이 아님. 다음은 입력 해상도(짧은 변 672px → 더 크게).
- wallplugs 입력 해상도 896px(짧은 변, 참고 16장으로 복귀): F1 공식 6.5/최적 21.6/AU-ROC 0.805, 672px 기준(6.0/24.0/0.804)과 사실상 동일. 전용 임계값·후처리 최선 24.2(672px 때 22.7과 비슷). 회전·뱅크 크기·해상도 세 레버 모두 AU-ROC를 0.80 언저리에서 못 벗어남 — 가설: 원통형 플러그의 시점에 따른 3D 외관 변화(그림자·하이라이트·원근)가 2D 특징의 정상 패치 거리에 노이즈 바닥을 만들어, 작은 결함 신호와 구분이 잘 안 됨. SOTA(SuperADD)도 wallplugs private F1 26.4로 8개 중 두 번째로 낮아 같은 해석과 일치. wallplugs에서 지금까지 얻은 가장 큰 개선은 임계값·후처리(6.0→22.7)뿐.

## 2026-11-06

- wallplugs 비교 방식 실험(HiResKNN pool/knn, 게임 모드, 변형당 약 2.5분): 3x3 이웃 묶기 AU-ROC 80.4→65.9로 크게 악화(작은 결함이 희석). 최근접 k개 평균은 k=9/25/50/100에서 AU-ROC 81.6/82.2/82.6/82.9, 최적 F1 27.1/29.2/31.0/33.3, 전용 임계값·후처리 최고 28.1/31.1/31.5/27.7. k=50 채택(실제 규칙 F1 최고). 층별(k=9): L6 54.6, L12 74.0, L18 81.6, L24 78.8, 층 빼기는 효과 없음(18+24 80.5, 12+18+24 81.2) → 4개 층 평균 유지. k는 공개 test로 골랐으므로 낙관적.
