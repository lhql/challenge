#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

package = Path(__file__).resolve().parents[1]
ws = package.parents[1]
saved = package / "results/preflight/src"
records = []
for snapshot in saved.rglob("*"):
    if snapshot.is_file():
        original = ws / "src" / snapshot.relative_to(saved)
        same = original.read_bytes() == snapshot.read_bytes()
        records.append(dict(path=str(original), unchanged=same))
(package / "results/preflight/asset_verification.json").write_text(
    json.dumps(records, indent=2)
)
if not all(r["unchanged"] for r in records):
    raise SystemExit("Previously existing asset changed; inspect before proceeding")
print(f"{len(records)} existing model/config files unchanged")
