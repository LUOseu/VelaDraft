#!/usr/bin/env python
"""Read-only audit of a prepared data directory (R1 migration check).

Scans ``<data-dir>/manifest.json`` plus the materialised media and reports:

* missing referenced files;
* manifest hash mismatches (file bytes changed after prepare);
* content-hash-name mismatches for files written by the post-R1 scheme
  (``{sample_id}_{sha256[:16]}{suffix}``), which catches tampering and
  foreign files at a claimed name;
* reference conflicts: one output file referenced with two different recorded
  hashes, the signature of the pre-R1 basename-collision data.

The audit never writes into the data directory. Exit code 0 = clean,
2 = issues found. The JSON report goes to stdout; pass ``--out`` to also
write it to a file.

    python scripts/audit_prepared_media.py --data-dir runs/<name>/data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash.utils import read_json, sha256_file  # noqa: E402

#: Post-R1 output name: ``{sample_id}_{16 hex content hash}{suffix}``.
CONTENT_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<hash>[0-9a-f]{16})(?P<suffix>\.[^.]*)?$")


def audit(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    report: dict = {
        "format": "vlm-dflash-media-audit-v1",
        "data_dir": str(data_dir),
        "issues": [],
        "num_manifest_rows": 0,
        "num_files_checked": 0,
    }
    issues: list[dict] = report["issues"]

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        issues.append({"kind": "missing_manifest", "path": str(manifest_path)})
        report["ok"] = False
        return report
    manifest = read_json(manifest_path)

    meta_path = data_dir / "meta.json"
    media_dir_name = read_json(meta_path).get("media_dir", "images") if meta_path.exists() else "images"
    media_dir = data_dir / str(media_dir_name)
    if not media_dir.is_dir():
        issues.append({"kind": "missing_media_dir", "path": str(media_dir)})
        report["ok"] = False
        return report

    referenced: dict[str, list[dict]] = {}
    files_checked: set[str] = set()
    rows = manifest.get("rows", [])
    report["num_manifest_rows"] = len(rows)
    for row in rows:
        row_id = row.get("id")
        names = list(row.get("images") or ([] if row.get("image") is None else [row["image"]]))
        hashes = list(row.get("image_sha256_list") or ([] if row.get("image_sha256") is None else [row["image_sha256"]]))
        if hashes and len(hashes) != len(names):
            issues.append(
                {
                    "kind": "reference_length_mismatch",
                    "id": row_id,
                    "images": len(names),
                    "hashes": len(hashes),
                }
            )
        for position, name in enumerate(names):
            recorded = hashes[position] if position < len(hashes) else None
            path = media_dir / name
            if not path.is_file():
                issues.append({"kind": "missing_file", "id": row_id, "file": name})
                continue
            files_checked.add(name)
            actual = sha256_file(path)
            referenced.setdefault(str(name), []).append({"id": row_id, "recorded": recorded})
            if recorded is not None and recorded != actual:
                issues.append(
                    {"kind": "manifest_hash_mismatch", "id": row_id, "file": name, "expected": recorded, "actual": actual}
                )
            match = CONTENT_NAME_RE.match(str(name))
            if match and match.group("hash") != actual[:16]:
                issues.append(
                    {
                        "kind": "content_hash_name_mismatch",
                        "file": name,
                        "name_hash": match.group("hash"),
                        "actual_hash": actual,
                    }
                )

    for name, refs in referenced.items():
        recorded = {entry["recorded"] for entry in refs if entry["recorded"]}
        if len(recorded) > 1:
            issues.append(
                {
                    "kind": "reference_hash_conflict",
                    "file": name,
                    "ids": sorted({entry["id"] for entry in refs}),
                    "hashes": sorted(recorded),
                }
            )

    report["num_files_checked"] = len(files_checked)
    report["ok"] = not issues
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="prepared data dir (data.jsonl/manifest.json/images)")
    parser.add_argument("--out", default=None, help="also write the JSON report here")
    args = parser.parse_args()
    report = audit(args.data_dir)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if report["ok"]:
        print(
            f"audit clean: {report['num_manifest_rows']} row(s), "
            f"{report['num_files_checked']} file(s) checked",
            file=sys.stderr,
        )
        return 0
    print(f"audit found {len(report['issues'])} issue(s)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
