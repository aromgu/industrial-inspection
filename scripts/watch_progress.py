"""Show the latest tqdm line of each running score.py log, refreshed in place.

    uv run python scripts/watch_progress.py            # all logs, Ctrl+C to quit
    uv run python scripts/watch_progress.py --once     # print once and exit
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

BAR = re.compile(r"\w+ (?:train|predict):\s+\d+%\|.*?\]")


def latest(log: Path) -> str:
    text = log.read_text(errors="ignore").replace("\r", "\n")
    bars = BAR.findall(text)
    return bars[-1] if bars else "(no progress bar yet)"


def snapshot(log_dir: Path) -> str:
    status = log_dir / "status.txt"
    lines = [f"{time.strftime('%H:%M:%S')}  finished: " + (", ".join(
        l.split()[0] for l in status.read_text().splitlines() if l.endswith("exit 0")) if status.exists() else "-")]
    for log in sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime):
        lines.append(f"[{log.stem.split('.')[0]}] {latest(log)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=float, default=5)
    args = p.parse_args()
    while True:
        out = snapshot(args.log_dir)
        print(out if args.once else f"\x1b[2J\x1b[H{out}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
