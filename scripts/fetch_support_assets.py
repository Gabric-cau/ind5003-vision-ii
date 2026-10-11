"""Fetch and checksum the original team's optional support archives."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/RX422040/vision-ii-penalty-analysis/releases/download/v2026.10.11-assets/"


def fetch(root: Path, download: bool = False) -> None:
    manifest = json.loads((root / "docs/git_supplement_manifest.json").read_text(encoding="utf-8"))
    records = manifest["archives"]
    print(json.dumps({"release": RELEASE, "total_bytes": sum(r["bytes"] for r in records),
                      "archives": records, "download_requested": download}, indent=2))
    if not download:
        return
    needed = sum(r["bytes"] for r in records)
    if shutil.disk_usage(root).free < needed + 256 * 1024**2:
        raise RuntimeError("Insufficient free space to download the support archives")
    for record in records:
        name = Path(record["path"]).name
        if record["path"] != f"project_files/{name}" or name not in (
                "visionii-support-01.zip", "visionii-support-02.zip"):
            raise ValueError("Unexpected support archive path")
        target = root / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with target.open("rb") as existing:
                digest = hashlib.file_digest(existing, "sha256").hexdigest()
            if target.stat().st_size == record["bytes"] and digest == record["sha256"]:
                print(f"Reusing verified archive: {name}")
                continue
            raise FileExistsError(f"Existing archive has different content: {target}")
        temporary = target.with_suffix(".zip.partial")
        if temporary.exists():
            raise FileExistsError(f"Interrupted download exists; inspect or remove it first: {temporary}")
        digest, count = hashlib.sha256(), 0
        request = urllib.request.Request(RELEASE + name, headers={"User-Agent": "VisionII-asset-fetcher"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                stream.write(block)
                digest.update(block)
                count += len(block)
        if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Downloaded archive checksum mismatch: {temporary}")
        temporary.rename(target)
        print(f"Downloaded and verified: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--download", action="store_true", help="Download approximately 3.58 GB")
    args = parser.parse_args()
    fetch(args.root, args.download)


if __name__ == "__main__":
    main()
