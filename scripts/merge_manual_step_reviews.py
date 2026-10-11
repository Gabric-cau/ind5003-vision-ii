"""Validate and merge frame-reviewed step counts with automated diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.annotation_provenance import is_human_verified
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "penalty_slicing_manifest.yaml"
DEFAULT_CLIPS_DIR = PROJECT_ROOT / "data" / "raw_clips"
DEFAULT_AUTO_CSV = PROJECT_ROOT / "data" / "processed" / "penalty_step_counts.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "penalty_step_counts_confirmed.csv"
DEFAULT_REVIEW_FILES = [
    PROJECT_ROOT / "data" / "annotations" / "penalty_step_manual_reviews.csv",
]

DEFINITION = (
    "After intentional run-up begins, count each foot transition from swing to a new "
    "landing; exclude a foot already planted at the boundary; include the final non-kicking "
    "support-foot plant before contact; exclude ball contact and recovery landings."
)

OUTPUT_FIELDS = [
    "clip_id",
    "match_key",
    "match_title",
    "penalty_number",
    "kicker",
    "team",
    "video_path",
    "confirmed_step_count",
    "confirmed_landing_frames",
    "manual_confidence",
    "manual_notes",
    "annotation_method",
    "candidate_step_count",
    "candidate_landing_frames",
    "annotator_type",
    "human_verified",
    "reviewer_id",
    "reviewed_at",
    "ground_truth_eligible",
    "step_definition",
    "auto_step_count",
    "auto_count_matches_manual",
    "auto_status",
    "auto_confidence",
    "auto_review_required",
    "auto_kick_frame",
    "auto_kicking_foot",
    "auto_step_frames",
    "auto_qa_warnings",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_rows(manifest_path: Path, clips_dir: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a mapping")

    rows: list[dict[str, Any]] = []
    for match_key, match in manifest.items():
        for penalty in match.get("penalties", []):
            number = int(penalty["number"])
            kicker = str(penalty["kicker"])
            clip_id = f"{match_key}_p{number:02d}_{kicker}"
            rows.append(
                {
                    "clip_id": clip_id,
                    "match_key": match_key,
                    "match_title": match.get("match_title", ""),
                    "penalty_number": number,
                    "kicker": kicker,
                    "team": penalty.get("team", ""),
                    "video_path": str(clips_dir / f"{clip_id}.mp4"),
                }
            )
    return rows


def _index_unique(rows: Iterable[dict[str, str]], source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        clip_id = str(row.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError(f"{source} contains a row without clip_id")
        if clip_id in indexed:
            raise ValueError(f"Duplicate clip_id in {source}: {clip_id}")
        indexed[clip_id] = row
    return indexed


def _normalise_confidence(value: str) -> str:
    confidence = value.strip().lower()
    aliases = {"h": "high", "m": "medium", "l": "low"}
    confidence = aliases.get(confidence, confidence)
    try:
        score = float(confidence)
    except ValueError:
        score = None
    if score is not None:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Numeric manual confidence must be in [0, 1]; got {value!r}")
        confidence = "high" if score >= 0.85 else "medium" if score >= 0.70 else "low"
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"Manual confidence must be high, medium, or low; got {value!r}")
    return confidence


def _optional_int(value: Any) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    return int(float(text))


def _normalise_landing_frames(value: str, expected_count: int, clip_id: str) -> str:
    frames = [int(item) for item in re.findall(r"\d+", value)]
    if len(frames) != expected_count:
        raise ValueError(
            f"Landing-frame count mismatch for {clip_id}: "
            f"expected {expected_count}, found {len(frames)} in {value!r}"
        )
    if any(frame < 0 for frame in frames) or any(
        current <= previous for previous, current in zip(frames, frames[1:])
    ):
        raise ValueError(f"Landing frames must be non-negative and strictly increasing: {clip_id}")
    return json.dumps(frames, separators=(",", ":"))


def merge(
    manifest_path: Path,
    clips_dir: Path,
    review_files: list[Path],
    auto_csv: Path,
    output_csv: Path,
) -> list[dict[str, Any]]:
    canonical = _canonical_rows(manifest_path, clips_dir)
    canonical_ids = {row["clip_id"] for row in canonical}

    manual_rows: list[dict[str, str]] = []
    for path in review_files:
        manual_rows.extend(_read_csv(path))
    manual = _index_unique(manual_rows, "manual review CSVs")

    missing = sorted(canonical_ids - set(manual))
    extra = sorted(set(manual) - canonical_ids)
    if missing or extra:
        raise ValueError(f"Manual labels do not match manifest; missing={missing}, extra={extra}")

    auto = _index_unique(_read_csv(auto_csv), str(auto_csv)) if auto_csv.is_file() else {}
    unexpected_auto = sorted(set(auto) - canonical_ids)
    if unexpected_auto:
        raise ValueError(f"Automated CSV contains non-canonical clip IDs: {unexpected_auto}")

    merged: list[dict[str, Any]] = []
    for base in canonical:
        clip_id = base["clip_id"]
        review = manual[clip_id]
        count = int(str(review.get("step_count", "")).strip())
        if count < 0:
            raise ValueError(f"Negative manual step count for {clip_id}")
        landing_frames = _normalise_landing_frames(
            str(review.get("landing_frames", "")).strip(), count, clip_id
        )
        confidence = _normalise_confidence(str(review.get("confidence", "")))
        automatic = auto.get(clip_id, {})
        auto_count = _optional_int(automatic.get("step_count"))
        verified = is_human_verified(review)
        row = {
            **base,
            "confirmed_step_count": count if verified else None,
            "confirmed_landing_frames": landing_frames if verified else "",
            "manual_confidence": confidence if verified else "",
            "manual_notes": str(review.get("notes", "")).strip(),
            "annotation_method": "human_frame_review" if verified else "unverified_candidate",
            "candidate_step_count": count,
            "candidate_landing_frames": landing_frames,
            "annotator_type": review.get("annotator_type", "unknown"),
            "human_verified": verified,
            "reviewer_id": review.get("reviewer_id", ""),
            "reviewed_at": review.get("reviewed_at", ""),
            "ground_truth_eligible": verified,
            "step_definition": DEFINITION,
            "auto_step_count": auto_count,
            "auto_count_matches_manual": (
                "" if auto_count is None or not verified else str(auto_count == count).lower()
            ),
            "auto_status": automatic.get("status", ""),
            "auto_confidence": automatic.get("confidence", ""),
            "auto_review_required": automatic.get("review_required", ""),
            "auto_kick_frame": automatic.get("kick_frame", ""),
            "auto_kicking_foot": automatic.get("kicking_foot", ""),
            "auto_step_frames": automatic.get("step_frames", ""),
            "auto_qa_warnings": automatic.get("qa_warnings", ""),
        }
        merged.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--auto-csv", type=Path, default=DEFAULT_AUTO_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-csv", action="append", type=Path, dest="review_files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_files = args.review_files or DEFAULT_REVIEW_FILES
    rows = merge(args.manifest, args.clips_dir, review_files, args.auto_csv, args.output_csv)
    low_confidence = sum(row["manual_confidence"] == "low" for row in rows)
    auto_matches = sum(row["auto_count_matches_manual"] == "true" for row in rows)
    print(f"Rows written: {len(rows)}")
    print(f"Manual low-confidence rows: {low_confidence}")
    verified_count = sum(row["ground_truth_eligible"] for row in rows)
    print(f"Human-verified rows: {verified_count}; unverified candidates: {len(rows) - verified_count}")
    print(f"Exact matches among verified rows: {auto_matches}/{verified_count} (not evaluated if zero)")
    print(f"Output: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
