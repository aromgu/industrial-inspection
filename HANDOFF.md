# 인수인계 — industrial-inspection (2026-11-11, PC 이전용)

새 PC의 새 세션에서 이 파일을 먼저 읽게 하세요. 아래 "⭐⭐⭐⭐ 새 PC 시작" 부분이 최신이고, 그 아래는 이전 기록입니다.

## ⭐⭐⭐⭐ 새 PC 시작 (2026-11-11)

### 옛 PC에서 지운 것 (새 PC에서 다시 받아야 함)
- 데이터셋 `datasets/MVTec_AD_2/` (61GB): https://www.mvtec.com/company/research/datasets/mvtec-ad-2 에서 다시 받아 같은 경로에 풀기. 라이선스 CC BY-NC-SA라 git에 올리지 않음.
- DINOv3 ViT-H+/16 가중치 `weights/dinov3_vith16plus.pth` (3.4GB): Meta DINOv3 다운로드 신청(https://ai.meta.com/dinov3/) 후 받기. 모델 코드는 `torch.hub` 캐시 `~/.cache/torch/hub/facebookresearch_dinov3_main`(github.com/facebookresearch/dinov3를 그 경로에 clone).
- SAM ViT-B `results/weights/sam_vit_b_01ec64.pth`: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- 옛 모델 체크포인트(`results/runs`, `results/onnx`, `results/*.pt`), 점수 지도(`results/maps`, `results/smoke`), 제출 파일(`results/submission*`)과 timm/HF 캐시의 DINOv2·WideResNet 가중치.

### git에 남긴 것
- 코드 `src/`, `scripts/`, `tests/`, 기록 `docs/experiment_log.md`, 그림 `docs/figures/`
- 작은 결과(CSV·로그·그림) `archive/results/` (옛 `results/`에서 지도·가중치 빼고 복사)

### 환경
- Python 3.12, `uv sync` (uv.lock). GPU 필요(DINOv3-H+ 1792px).
- 한글 경로 주의: anomalib은 비ASCII 절대경로 거부 → 데이터 경로는 레포 기준 상대경로.

### 최종 wallplugs 모델 다시 만들기
1. 점수 지도 (명령은 기록에서 재구성한 것 — 먼저 `--smoke`로 확인하고 값 범위를 볼 것):
   `.venv/Scripts/python.exe scripts/hires.py --categories wallplugs --backbone dinov3_vith16plus --layers 15 23 --tile 640 --overlap 128 --bank all --brightness 0.8 1.2 --knn 1 --sigma 0 --short-side 1792 --name s0_r1792_L1523 [--private]`
2. 평가: `scripts/wp_eval.py --model s0_r1792_L1523 --guided 4 0.01 --seed-gauss --qs 0.95` → 공개 test F1 73.4 (개발 70.5 / 평가 76.3), 오경보 5%
3. 형태 복원: SAM 플러그 마스크 `scripts/plugs_knn.py --category wallplugs --split <split> --cache-only` 후 `scripts/make_submission.py --complete` → 공개 F1 75.7
4. 그림: `wp_cut_gallery.py`, `wp_complete_steps.py`, `wp_private_results.py`, `wp_model_diagram.py` 등 (`docs/figures/`)

### 결과 요약 (wallplugs, 공식 비공개 test, 2026-11-09 제출 "DINOv3-HiResKNN")
| | 우리 | SuperADD | 리더보드 최고 |
|---|---|---|---|
| SegF1 | 33.6 | 26.4 (논문 Table 2) | 51.9 |
| AucPro_0.05 | 70.6 | 48.3 | 80.7 |
| ClassF1 | 69.0 | 54.0 | 81.2 |
- 공개 → 비공개 SegF1 75.7 → 33.6 (SuperADD 79.2 → 26.4). 다른 7개 카테고리는 튜닝 안 함(제출용).
- 제출: benchmark.mvtec.com, 7일에 2회, 초안 없음. 11/9에 1회 사용. **업로드는 사용자가 직접**, 제출 관련은 먼저 물어볼 것.

### 11/9~11/10에 한 것
- MuSc식 상호 비교(`scripts/wp_musc.py`, test 사진끼리 비교, k=20): shift 조명 노이즈 제거. 단독 F1 71.0, 기존 지도와 반반 섞으면 **75.3 (개발 72.5 / 평가 78.0)**, 오경보 8.3%. 형태 복원은 아직 안 얹음. test를 모아 보는 transductive 방식이라 규칙 확인 전까지 제출엔 안 씀. fp16 거리 계산은 넘쳐서 0이 됨 → fp32 cdist 사용.
- 잘린 플러그 예측 그림 `docs/figures/wp_cut_gallery.png`, 슬라이드용 `wp_cut_pair_clean.png`: 000번은 보통·과·저노출에서 7% → 94%, shift 3장과 001번은 검출 없음.
- 포트폴리오: 노션 "[포트폴리오] wallplugs 결함 검출: 방법과 결과" https://app.notion.com/p/4133a009bcdf81da96dcdf18180542fe , 2장 슬라이드 https://claude.ai/artifact/Ya5hKmYfFxqurAvhyM9FYU (사용자 PPT는 `Desktop/2026 구아롬 포트폴리오 vision.pdf`; 오타·7페이지 Avg 확인 필요).
- 설명용 사실: 1792px는 원본 2448×2048을 짧은 변 88%로 줄인 것(SuperADD 1280), 640px 타일 128 겹침. 사진별 보정은 평균이 아니라 p90을 뺌(씨앗에만). 이중 임계값은 고전 히스테리시스. 분위수(95%, 99.5%)는 공개 test에서 골랐고 개발/평가 분할로 확인.

### 다음 후보
- 가짜 결함으로 분위수 자동 선택(`wp_synth.py`) → 손으로 고른 95%/99.5%와 비교 (사용자에게 제안만 함)
- MuSc 섞기 + 형태 복원, 비공개용 MuSc 계산(규칙 확인 후, 약 1~1.5시간)
- 실행 시간·메모리 측정

### 작업 습관 (사용자 선호)
- 긴 실행 전 스모크 테스트, 출력 값까지 확인 / 모든 학습·평가는 PNG 그래프 저장
- 설명은 한국어로 쉽게, 담백하게. 사용자가 게임 중이면 `--game-mode`
- 노션 그림은 `notion-create-attachment`에 GitHub raw URL(커밋 SHA 고정)

---

## 경로

- **프로젝트**: `C:\Users\살붕이\Documents\GitHub\industrial-inspection`
- **GitHub**: https://github.com/aromgu/industrial-inspection (origin, master에 push)
- **데이터셋**: `datasets/MVTec_AD_2/<category>/...` (MVTec 공식 데이터, test_private 포함)
- **DINOv3-H+ 가중치**: `weights/dinov3_vith16plus.pth` (3.4GB, git 제외). 모델 코드는 torch.hub 캐시 `~/.cache/torch/hub/facebookresearch_dinov3_main`
- **SuperADD 코드**: `C:\Users\살붕이\Documents\GitHub\SuperADD` (참고용, 자체 venv)
- **Notion**: 모델 상세 https://app.notion.com/p/4103a009bcdf818faa4ffc0a439464c2 , 계획·진행 https://app.notion.com/p/40f3a009bcdf8133ad55f6d6d6e7f66d , 인수인계 https://app.notion.com/p/40c3a009bcdf819ca8ebd4fb888f7e56

## ⭐⭐⭐ 2026-11-09 상태 (최신)

- **제출 파일 준비 완료(업로드 안 함)**: `results/submission.tar.gz`(기본), `results/submission_complete.tar.gz`(+ wallplugs 형태 복원, 추천). 둘 다 `make_submission.py` 검증 통과. 업로드는 사용자가 https://benchmark.mvtec.com 에서(7일 2회).
- 3번 형태 복원(씨앗 기반 연장, `industrial_inspection.complete`): 공개 F1 73.4 → 75.7, 큰 결함 77→87%, 오경보 5% 유지. 비공개 플러그 마스크 캐시 `results/plugs_cache/wallplugs/wallplugs/test_private(_mixed)`.
- 1번 대조 학습(`wp_adapt.py`, 맵 `s0_r1792_adapt_inv`): shift 노이즈 제거되지만 작은 결함도 약해져 F1 66~69. 2번 확대 재검사(`wp_zoom2.py`): 실패.
- 11/9 오전 추가: CFA·조명만 불변 대조 학습, 90도 회전 사본 뱅크 모두 최종(73.4) 못 넘음. shift 노이즈는 학습·검증에 없는 조명 도메인이 원인(조건을 안다고 가정한 상한도 73.6). 돌고 있는 작업 없음.
- Notion: 11/9 페이지 https://app.notion.com/p/4123a009bcdf8166b90ee50f71c8dccb

## ⭐⭐ 2026-11-08 상태

- 최종 설정: `wp_eval.py --model s0_r1792_L1523 --guided 4 0.01 --seed-gauss --qs 0.95` → 작은 50%, 큰 77%, 오경보 5%, F1 73.4 (개발 70.5 / 평가 76.3).
- 오늘 한 것과 결과: Notion 결과 보고서 https://app.notion.com/p/4123a009bcdf8128aecaee78d4f09405 , 계획서 https://app.notion.com/p/4123a009bcdf8129ba00c76d0fad56a0 , docs/experiment_log.md 맨 아래.
- 효과: 큰 뱅크(노이즈↓), 언샤프 입력(작은 결함↑), 유도 필터(범위↑), 앙상블(오경보↓). 실패: CLAHE 단독, 패딩, 조명 증강 뱅크, 대칭, Cutout 판별기(P5 중단).
- 돌고 있는 작업 없음. 큰 뱅크+언샤프 조합은 shift 오경보가 커져 나빠짐(F1 61~65). 씨앗 퍼짐 정규화(`--seed-scale 50`)는 최종과 비김.
- **정체**: F1 약 73 / 작은 결함 50%에서 모든 조합이 멈춤. 병목 = 아주 작은 결함 5개 부품(000·001·002·004·007, 신호 < 노이즈 바닥) + shift 조명 노이즈(검증에 없음, 증강으로 재현 못 함).
- 11/8 오후 추가: 직선 잘림 가짜 결함 판별기(`wp_synth.py --kinds 6`, 맵 `synth_cut6`) → 000의 부러진 단면은 잡지만 GT(빈 공간)를 못 덮어 0/6, 결합 시 오경보 3.3%·F1 72.5. 히스토그램 매칭(`hires.py --histmatch`, 맵 `s0_r1792_hm`) → shift 노이즈↓ 결함↓, F1 73.9 동률. 잘린 결함은 "단면 씨앗 + 플러그 길이 복원"이 다음 시도.
- 다음 후보(접근 변경): shift 조명 정상 사진 확보, DINOv3를 정상 사진으로 미세조정(대조 학습 등), 비공개 test에 최종 설정 적용 → 제출 파일(제출은 사용자 확인 후).

## ⭐ 할 일 (2026-11-07 밤 사용자 지시로 멈춤, 여기서부터 이어서)

돌아가는 작업 없음. wallplugs 실험 페이지: https://app.notion.com/p/4113a009bcdf81d3a24ded5b4f94a756 (맨 위 "지금 상황", "계획")

**현재 최고**: DINOv3-H+ 1792px, 층 15·23(0-based, "16·24번째"), k=1, σ=2 맵 `s0_r1792_L1523` + 사진별 오프셋(사진 자신의 p90을 뺌) 씨앗 + 이중 임계값(`scripts/wp_hyst.py --seed r1792L_s2_iq90 --extent s0_r1792_L1523 --sigma 2 --qs 0.9 --qe 0.995`). 작은 결함 51.7%, 큰 결함 76.7%, 정상 사진 오경보 6.7%, F1(전체) 70.9, 평가용 절반 74.7. 평가는 `scripts/wp_local.py`, `wp_hyst.py`, `wp_sweep.py`.

문제를 P1~P5로 나눔: P1 촬영 조건 오경보(해결), P2 결함 범위(해결), P3 아주 작은 결함 002·004·007, P4 형태 변형 005, P5 없어진 부분 001.

1. [ ] **P3-1 큰 메모리 뱅크** — 멈춘 실행을 처음부터 다시:
   `.venv/Scripts/python.exe scripts/hires.py --categories wallplugs --backbone dinov3_vith16plus --layers 15 23 --tile 640 --overlap 128 --bank all --brightness 0.8 1.2 --knn 1 --sigma 0 --short-side 1792 --name s0_r1792_bb --per-image-cap 6000 --max-bank 300000 [--game-mode]`
   (게임 모드 약 1시간, 아니면 약 35분. 패치 더미는 CPU에 두도록 고쳐서 게임 모드 VRAM 제한에서도 돌아감.) 끝나면 `wp_cascade.py --norm iq90 ... --name bb_iq90`로 씨앗 맵을 만들고 `wp_hyst.py --seed bb_iq90 --extent s0_r1792_bb --sigma 2`로 비교. 확인할 것: 정상 사진 최고점(가장자리 제외)이 내려갔는지, 002·004·007이 잡히는지(`scripts/wp_fp_where.py`의 분석을 새 맵으로).
2. [ ] P3-2 효과 있으면 사진당 패치 전부 / 뱅크 50만 이상 (RAM 31GB 주의: 사진당 6,000 x 256장 x 2층 ≈ 8GB).
3. [ ] P3-3 대비 강화(CLAHE) 입력 (사용자 제안) — hires.py에 전처리 옵션 추가 필요, 뱅크·test 모두 같은 전처리.
4. [ ] P3-4 k=3~5 (`--knn 5 --k1-name ...`로 한 번에).
5. [ ] P4 형태 변형: `scripts/wp_synth.py`에 날개·홈 휨형 가짜 결함 추가 → 거리 맵과 결합, 씨앗 임계값 따로.
6. [ ] P5 기울기 보정 실루엣(2~3시간). 3D 렌더링 비교는 이번에는 안 함.
7. [ ] 마무리: 개발/평가 검증, 그림 갱신(`scripts/wp_report_figs.py`), 비공개 test 적용 → 제출 파일(제출은 사용자 확인 후), README·Notion 정리.

**이미 해 보고 안 된 것** (다시 하지 말 것): 2048px, TTA, 확대 2단계(wp_zoom), top-hat 씨앗, 단일 층 씨앗, 같은 사진 내 비교(self), 지역 배경 빼기(median), 사진별 p99 오프셋, 게이트(wp_gate), 소벨 재순위(wp_sobel, wp_sobel_synth, wp_rerank — 사진 안 순위만 오르고 F1 그대로), 길이·실루엣 형태 검사, 가짜 결함 판별기 단독/결합(wp_synth), 플러그 단위 DINOv3 비교(너무 느림, 검증 33장 56분).

**알아둘 것**: `r1792L_s2_iq90` 같은 wp_cascade 출력은 검증 최대값 위가 잘려 있음(qmatch) — 임계값 판단에는 문제없지만 봉우리 분석에는 `m - percentile(m, 90)`을 직접 쓸 것. 공개 test 정상 60장 중 shift 조건 30장의 원래 맵 점수가 통째로 높음(검증은 보통 조명뿐).

## 지금 상태 (한 줄, 2026-11-07 오후)

**wallplugs 한 카테고리에 집중.** 학습 없는 메모리뱅크 방식(HiResKNN)에서 백본을 DINOv3-H+로 바꾸자 wallplugs AU-ROC 82.6 → 90.6, 합친 F1 36.7 → 71.2(SuperADD 공개 79.2). 이후 위 "할 일"의 현재 최고까지 진행.

## 방향 (사용자 결정, 2026-11-07)

- 8개 카테고리 전체 성능은 벤치마크 제출용일 뿐, **연구는 wallplugs 하나에 집중**. 시간이 많지 않음.
- 제출(benchmark.mvtec.com)은 7일에 2번, 초안 없음 → 제출 전 반드시 사용자 확인. 아직 제출 안 함.

## 모델 (학습 없음)

1. 얼린 백본(DINOv2-L 또는 DINOv3-H+)으로 사진을 겹치는 조각으로 나눠 패치 특징을 뽑음
2. 정상 학습 사진의 패치 특징을 메모리 뱅크에 저장
3. test 패치마다 뱅크의 가장 가까운 정상 패치(k=1 또는 k=50 평균)까지 거리 = 이상 점수
4. 정상 검증 사진 점수로 임계값 결정, 후처리(닫힘·채우기·작은 덩이 제거)

- 코드: `src/industrial_inspection/hires.py`(HiResKNN, `_DinoV3` 어댑터), `scripts/hires.py`(`--backbone --layers --tile --overlap --bank all --brightness --knn --k1-name`)
- wallplugs 전용: SAM 플러그 분리 비교 `scripts/plugs_knn.py`, 결합 `scripts/plugs_fuse.py` (아직 DINOv2-L 기반)
- 평가: `scripts/seg_eval.py`(F1, AU-ROC), `scripts/seg_post.py`(임계값·후처리 탐색)
- 제출 파일: `scripts/make_submission.py` (CONFIG가 아직 DINOv2 기반 구성. 비공개 wallplugs 플러그 맵 미완성)

## 핵심 결과

| | 결과 |
|---|---|
| 우리 제출 후보 (DINOv2-L) 8개 평균 F1 | 45.0 (SuperADD 62.6) |
| 격차 원인 시험 | 후처리 아님, 뱅크 구성 아님, 입력 크기 아님, **백본이 원인** |
| DINOv3-H+ vial·fruit_jelly·rice | 평균 63.5 (DINOv2-L 48.3, SuperADD 64.3) |
| DINOv3-H+ wallplugs k=1 | AU-ROC 90.6, F1 71.2 (SuperADD 임계값 규칙), 사진별 검출 23% |
| DINOv3-H+ can, sheet_metal, walnuts | 맵만 저장(`results/maps/hires_dv3_*`), 평가 안 함. fabric은 안 돌림 |

자세한 기록: `docs/experiment_log.md`, README "SuperADD와의 격차는 어디서 오나", "wallplugs: DINOv3-H+".

## 다음 할 일 (wallplugs)

1. DINOv3 맵(`hires_dv3_k1__wallplugs`) 위에서 작은 결함용 임계값/후처리를 **검증 데이터만으로** 다시 고르기
2. 플러그별 점수 정규화 (결함 위치가 주변 최고점인데 임계값 미달인 문제)
3. SAM 플러그 비교(`plugs_knn.py`)를 DINOv3로 바꿔 다시 결합 (SAM 캐시는 `results/plugs_cache/wallplugs/`에 있음)
4. 그래도 부족하면: 가짜 결함 합성 + 작은 분할 머리(GLASS/SimpleNet 계열), 같은 사진 안 플러그끼리 비교(MuSc 계열)

## 작업 습관 (사용자 선호)

- 긴 실행 전 반드시 작은 스모크 테스트 (`--smoke`)
- 학습/평가마다 PNG 그래프 저장
- 사용자가 게임(메이플스토리)·영화 중이면 `--game-mode`로 실행. 아니면 전체 속도
- 결과는 Notion에도 정리. 그림은 Notion에 파일로 직접 업로드(`notion-create-attachment`의 source_url = GitHub raw URL, 커밋 SHA 고정)

## 알려진 함정

1. **한글 경로**: anomalib은 비ASCII 절대경로를 거부 → 데이터 경로는 레포 기준 상대경로
2. **pandas<3 고정**: anomalib 2.7과 pandas 3 비호환
3. **점수 후처리 끄기**: `score.py`는 `post_processor=False`
4. **합친 픽셀 F1의 함정**: 큰 결함 몇 장이 점수를 좌우함. 사진별 F1도 같이 볼 것
5. `nohup ... &`를 백그라운드 실행 안에 쓰지 말 것 (작업이 끝난 걸로 잘못 보고됨)
