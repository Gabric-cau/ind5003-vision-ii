import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from scripts.run_fresh_scoreboard_experiment import main


def test_full_source_window_stops_before_next_canonical_attempt(tmp_path):
    originals = tmp_path / "originals"
    clips = tmp_path / "clips"
    originals.mkdir()
    clips.mkdir()
    original = originals / "source.avi"
    writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    assert writer.isOpened()
    for index in range(200):
        writer.write(np.full((120, 160, 3), (20, 60 + index % 20, 25), np.uint8))
    writer.release()
    video_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    sources = []
    for index, (start, end) in enumerate(((1., 2.), (5., 6.)), 1):
        context = clips / f"clip{index}.avi"
        context.write_bytes(original.read_bytes())
        sources.append({"clip_id": f"clip{index}", "event_id": index, "match_id": "test",
                        "penalty_number": index, "fps": 30., "source_start_frame": 0,
                        "source_filename": original.name, "source_video_sha256": video_hash,
                        "clip_filename": context.name, "clip_sha256": video_hash,
                        "frame_count": 200, "slice_start_sec": start, "slice_end_sec": end,
                        "original_slice_end_frame_in_clip": int(end * 30)})
    (clips / "source_manifest.json").write_text(json.dumps({"clips": sources}))
    output = tmp_path / "output"
    assert main(["--clips-dir", str(clips), "--original-videos-dir", str(originals),
                 "--output-dir", str(output)]) == 0
    first = json.loads((output / "clip1.json").read_text())
    second = json.loads((output / "clip2.json").read_text())
    assert first["search_interval_frames"] == [60, 150]
    assert all(60 <= row["source_frame_id"] < 150 for row in first["observations"])
    assert second["search_interval_frames"] == [180, 200]
    assert first["source_video_sha256"] == video_hash
    assert first["frame_index_basis"] == "original_full_video"
    assert first["scoreboard_outcome"]["rule_outcome"] == "unknown"
    assert first["after_bound_is_semantic_event_end_truth"] is False
    assert first["reference_labels_used_for_prediction"] is False
    assert json.loads((output / "scoreboard_protocol.json").read_text())["frozen_before_extraction"] is True


def test_direct_scoreboard_cli_help():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts/run_fresh_scoreboard_experiment.py"), "--help"],
                               capture_output=True, text=True, check=True)
    assert "--original-videos-dir" in completed.stdout
