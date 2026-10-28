# 인수인계서 (2026-10-28 작업 중단 시점)

이 컴퓨터(Ryzen 5 5600 / RAM 32GB / RTX 3050 8GB, Windows)에서 진행하던 MVTec AD 2 베이스라인 실험을 사용자 요청으로 중단한 시점의 상태 기록. 실험 결과 자체는 [docs/experiment_log.md](experiment_log.md)에 있고, 여기는 **"지금 뭐가 돌아가고 있고, 어떻게 이어서 하면 되는지"**에 집중.

## 1. 환경 세팅 (완료됨, 다시 안 해도 됨)

- `pyproject.toml`의 torch 인덱스를 `cu128` → `cu126`으로 바꿔둠. 이 머신 드라이버(560.94)가 CUDA 12.6까지만 지원해서 cu128은 `torch.cuda.is_available()`이 False가 됨. **다른 머신에서 이 코드를 받으면 드라이버의 `nvidia-smi` 상단에 찍히는 CUDA Version을 보고 인덱스를 다시 맞춰야 할 수 있음.**
- Windows **Smart App Control**이 uv가 받아온 서명되지 않은 Python 실행을 차단해서 꺼둠 (설정 > 개인정보 보호 및 보안 > Windows 보안 > 앱 및 브라우저 제어). 이미 꺼져 있으면 신경 안 써도 됨.
- 데이터셋은 `D:\datasets\MVTec_AD_2`에 있음 (캐시는 `D:\datasets\.hf_cache`). 8개 카테고리 전부, train/validation/test_public 다 있음. **다시 받을 필요 없음.**
- `uv sync` 이미 되어 있음 (`.venv` 존재). 새로 클론한 거면 `uv sync` 한 번 필요.

## 2. 코드에 넣은 수정 (커밋됨, origin/master에 푸시 완료)

- `scripts/score.py`:
  - `Engine(..., enable_progress_bar=False, ...)` — Lightning 기본 Rich 진행률 표시줄이 한글 Windows 콘솔(cp949)에서 유니코드(•) 인코딩 실패로 크래시하던 문제. **이 줄 지우면 다시 크래시함.**
  - `torch.set_float32_matmul_precision("high")` — RTX 3050 Tensor Core 활용.
  - `--train-batch-size`, `--precision` 옵션 추가 (더 작은 GPU에서 오버라이드용. 지금은 안 씀 — Dinomaly-L 기본 배치 4가 8GB에서 smoke 테스트 통과해서 오버라이드 불필요했음).
- `scripts/sweep.sh`:
  - `python -m uv run python ...` → `uv run python ...`로 고침. 원래 코드는 전역 Python에 `uv` 모듈이 없어서 **모든 카테고리가 1초 안에 즉시 실패(exit 1)**하던 버그였음. (sweep_status.txt에 남아있는 15:07 근처의 exit 1 기록들이 이 버그 때문 — 실제 실패 아님, 무시해도 됨.)
  - 데이터셋 루트를 넘길 수 있게 3번째 인자 `root` 추가: `sh scripts/sweep.sh "모델들" "카테고리들" "D:/datasets/MVTec_AD_2"`. **루트 안 넘기면 기본값(`datasets/MVTec_AD_2`, 레포 안)을 찾다가 실패함.**

## 3. 지금 결과가 있는 것 / 없는 것

`results/scores/*.csv` 기준 (전체 8개 카테고리: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts):

| 모델 | 완료 카테고리 | 남은 카테고리 |
|---|---|---|
| PatchCore | 8/8 (전부) | 없음 |
| MH-PatchCore | 8/8 (전부) | 없음 |
| Dinomaly (ViT-B) | 4/8: can, fabric, fruit_jelly, rice | **sheet_metal, vial, wallplugs, walnuts** |
| Dinomaly-L (ViT-L) | 0/8 | 전부 |
| EfficientAD | 1/8: sheet_metal | **can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts** |

- Dinomaly `sheet_metal`은 학습 중(약 93% 지점 추정)에 강제 종료해서 **CSV 파일이 없음** (score.py는 끝까지 가야 CSV를 씀). 처음부터 다시 돌아감.
- 모든 프로세스(python, sh)는 작업 중단 시점에 `taskkill /T /F`로 정리했음. 지금 GPU는 완전히 비어 있음.

## 4. 다시 시작하는 법

```sh
cd industrial-inspection

# 1) Dinomaly 남은 4개 카테고리
sh scripts/sweep.sh "dinomaly" "sheet_metal vial wallplugs walnuts" "D:/datasets/MVTec_AD_2"

# 2) Dinomaly-L 전체 8개 (smoke 테스트로 8GB에서 OOM 없음 확인됨, 기본 배치 4 그대로)
sh scripts/sweep.sh "dinomaly_l" "can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts" "D:/datasets/MVTec_AD_2"

# 3) EfficientAD 남은 7개 카테고리 (제일 오래 걸림, 카테고리당 ~2시간)
sh scripts/sweep.sh "efficientad" "can fabric fruit_jelly rice vial wallplugs walnuts" "D:/datasets/MVTec_AD_2"
```

`sweep.sh`는 resumable — `results/scores/`에 이미 CSV가 있는 (모델, 카테고리) 조합은 자동으로 건너뜀. 그냥 전체 8개 카테고리를 다시 넣어도 안전함.

**주의**: GPU가 1장(8GB)뿐이라 위 3개를 동시에 돌리면 안 됨. 하나씩 순서대로.

### 속도 참고 (이 머신, RTX 3050 실측)

| 모델 | 속도 | 카테고리당 시간 |
|---|---|---|
| PatchCore / MH-PatchCore | - | ~4~5분 |
| Dinomaly (ViT-B) | ~1.3~1.4 it/s | ~55~65분 |
| Dinomaly-L (ViT-L) | 미측정 (smoke만 함) | 배치 4 기준 Dinomaly보다 느릴 것으로 예상 |
| EfficientAD | ~9~10 it/s (안정화 후) | ~2시간 |

### 평가/정리

```sh
uv run python scripts/evaluate.py results/scores/*.csv
```

`results/report.csv`와 `results/figures/`에 비교 그래프가 생성됨. [docs/experiment_log.md](experiment_log.md)에 모델별 표로 정리해뒀으니, 새 카테고리 끝날 때마다 같은 형식으로 추가하면 됨.

## 5. 보류 중인 사용자 요청

- **Notion 연동**: `claude mcp list`로 확인해보면 Notion MCP(`https://mcp.notion.com/mcp`)는 이미 연결돼 있음. 근데 **이 세션이 시작된 뒤 연결된 거라 도구가 로드 안 됨** — 새 Claude Code 세션을 열면(창 재시작) Notion 도구가 바로 보일 것. 그러면 사용자가 공유한 페이지(`industrial-inspection MVTec AD 2 sheet_metal`)에 실험 로그를 이어서 올리면 됨.
- 사용자가 "학습할 때마다/완료될 때마다 알려달라"고 했음 — `Monitor`로 `results/logs/sweep_status.txt`를 `tail -f`해서 `exit`/`done` 라인을 잡는 방식을 썼음 (30분마다 만료돼서 재가동 필요). 이어서 할 때 같은 패턴 쓰면 됨:
  ```sh
  tail -n 0 -f results/logs/sweep_status.txt | grep --line-buffered -E "exit|done"
  ```

## 6. 알아두면 좋은 것

- `D:\datasets\.hf_cache`에 huggingface_hub 캐시가 그대로 있음 — 데이터셋 다시 받을 일 생겨도 캐시 덕분에 빠름.
- `results/`는 `.gitignore`에 있어서 CSV/로그/그림은 git에 안 올라감. 사람이 읽는 요약만 `docs/experiment_log.md`에 수동으로 옮겨서 커밋.
- README의 "베이스라인 실험" 표(사용자 머신 RTX 5060 Ti 추정)와 이 머신 결과를 비교하면 sheet_metal 쪽 수치가 꽤 근접함 (PatchCore 0.712 vs 0.739, MH-PatchCore 0.650 vs 0.629, EfficientAD 0.598 vs 0.573) — 파이프라인이 머신이 바뀌어도 일관되게 동작한다는 뜻으로 해석하면 됨.
