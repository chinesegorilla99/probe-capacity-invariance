"""Result-artifact manifest — SHA-256 lineage for every checkpoint and stack.

Records path, size, and SHA-256 for the artifacts the analysis consumes
(encoder checkpoints, probe stacks/meta, reports, figures, configs) so any
regeneration can be verified byte-for-byte and a corrupt or stale file is
caught before it enters the stats. Synthetic fixtures are included but tagged.

    python -m src.eval.manifest                 # write results/MANIFEST.json
    python -m src.eval.manifest --check         # verify against the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GLOBS = (
    "results/encoders/**/*.pt",
    "results/probes/**/stacks.npz",
    "results/probes/**/meta.json",
    "results/probes/**/hypothesis_report.json",
    "results/probes/hypotheses.json",
    "results/probes/pixel_*.json",
    "results/figures/**/*.csv",
    "configs/**/*.yaml",
)
MANIFEST = Path("results/MANIFEST.json")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect(root: Path) -> dict:
    entries = {}
    for pattern in GLOBS:
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            entries[rel] = {
                "sha256": _sha256(p),
                "bytes": p.stat().st_size,
                "synthetic": "_synthetic" in rel,
            }
    return entries


def write(root: Path) -> None:
    entries = collect(root)
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(entries),
        "files": entries,
    }
    out = root / MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {out} ({len(entries)} files)")


def check(root: Path) -> int:
    out = root / MANIFEST
    if not out.exists():
        print(f"[manifest] MISSING {out} — run without --check first")
        return 1
    recorded = json.loads(out.read_text())["files"]
    current = collect(root)
    changed = [p for p in recorded if p in current
               and current[p]["sha256"] != recorded[p]["sha256"]]
    missing = [p for p in recorded if p not in current]
    added = [p for p in current if p not in recorded]
    for label, paths in (("CHANGED", changed), ("MISSING", missing), ("ADDED", added)):
        for p in paths:
            print(f"[manifest] {label} {p}")
    ok = not (changed or missing)
    print(f"[manifest] {'OK' if ok else 'MISMATCH'}: "
          f"{len(recorded)} recorded, {len(changed)} changed, "
          f"{len(missing)} missing, {len(added)} new (new files are informational)")
    return 0 if ok else 1


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--check", action="store_true", help="verify instead of write")
    args = ap.parse_args()
    root = Path(args.root)
    raise SystemExit(check(root) if args.check else write(root))


if __name__ == "__main__":
    _main()
