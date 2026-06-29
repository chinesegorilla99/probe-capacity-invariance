"""Append-only JSONL logger for per-epoch training/eval records."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """One JSON object per line. Flushes each write so runs are crash-safe."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")

    def log(self, **record: Any) -> None:
        record.setdefault("wall_time", time.time())
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
