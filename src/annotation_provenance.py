"""Explicit provenance checks; a filename or confidence is not human verification."""

from datetime import datetime
from typing import Any, Mapping


def is_human_verified(row: Mapping[str, Any]) -> bool:
    if str(row.get("human_verified", "")).lower() not in {"true", "1"}:
        return False
    if row.get("annotator_type") != "human" or not str(row.get("reviewer_id", "")).strip():
        return False
    try:
        datetime.fromisoformat(str(row.get("reviewed_at", "")))
    except ValueError:
        return False
    return True


def require_verified(rows, label: str, allow_unverified: bool = False) -> None:
    unverified = [row.get("clip_id", "unknown") for row in rows if not is_human_verified(row)]
    if unverified and not allow_unverified:
        raise ValueError(
            f"{label}: {len(unverified)} unverified annotations. "
            "Legacy AI estimates are not human ground truth. "
            "Use --allow-unverified-annotations only for explicitly labelled exploratory runs."
        )
