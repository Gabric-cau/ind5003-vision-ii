"""Safely append the final Vision II step-counting workflow to the project notebook.

The updater is deliberately idempotent: it creates a one-time backup, repairs
only four exact legacy cells identified by their stable notebook IDs, and
appends each final section only when its stable cell marker is absent.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "vision2_penalty_analysis.ipynb"
BACKUP_PATH = PROJECT_ROOT / "notebooks" / "archive" / "vision2_penalty_analysis_before_step_counting.ipynb"
SECTION_TAG = "vision-ii-final-step-counting"
CONFIRMED_MARKER_ID = "vision2-steps-confirmed-title"


def _source(text: str) -> list[str]:
    """Return a Jupyter-compatible list of source lines."""

    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def _markdown(cell_id: str, text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"tags": [SECTION_TAG]},
        "source": _source(text),
    }


def _code(cell_id: str, text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": [SECTION_TAG]},
        "outputs": [],
        "source": _source(text),
    }


def _find_cell(notebook: dict[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one cell with id {cell_id!r}; found {len(matches)}"
        )
    return matches[0]


def _replace_legacy_motion_call(
    notebook: dict[str, Any],
    *,
    cell_id: str,
    result_name: str,
    track_name: str,
    kick_frame: int,
) -> bool:
    """Upgrade one exact 1D kicking-foot call to the current 2D pose API."""

    cell = _find_cell(notebook, cell_id)
    old_source = "".join(cell.get("source", []))
    if "left_ankles=" in old_source and "right_ankles=" in old_source:
        return False

    required_fragments = (
        "classify_kicking_foot_from_motion(",
        "left_ankle_x=left_ankle_x",
        "right_ankle_x=right_ankle_x",
        "left_hip_x=left_hip_x",
        "right_hip_x=right_hip_x",
    )
    if not all(fragment in old_source for fragment in required_fragments):
        raise RuntimeError(
            f"Legacy cell {cell_id!r} no longer matches the expected safe rewrite"
        )

    cell["source"] = _source(
        f"""
        {result_name} = extractor.classify_kicking_foot_from_motion(
            frames=frames,
            left_ankles=[{track_name}[fid]["keypoints"][15] for fid in frames],
            right_ankles=[{track_name}[fid]["keypoints"][16] for fid in frames],
            left_hips=[{track_name}[fid]["keypoints"][11] for fid in frames],
            right_hips=[{track_name}[fid]["keypoints"][12] for fid in frames],
            kick_frame={kick_frame},
            window=3,
        )

        {result_name}
        """
    )
    cell["execution_count"] = None
    cell["outputs"] = []
    return True


def _repair_legacy_cells(notebook: dict[str, Any]) -> list[str]:
    repaired: list[str] = []
    repairs = (
        ("c5bb68692bd1bc2c", "result", "tracked", 48),
        ("a8bea5b5edebe629", "kane_result", "tracked", 71),
        ("fc79b56d98034d8f", "pivaric_result", "tracked_pivaric", 142),
    )
    for cell_id, result_name, track_name, kick_frame in repairs:
        changed = _replace_legacy_motion_call(
            notebook,
            cell_id=cell_id,
            result_name=result_name,
            track_name=track_name,
            kick_frame=kick_frame,
        )
        if changed:
            repaired.append(cell_id)

    model_cell = _find_cell(notebook, "e0bb2d9c2245eca7")
    model_source = "".join(model_cell.get("source", [])).strip()
    if model_source == "pose_model_m\nimgsz=1280":
        model_cell["source"] = _source(
            """
            # The following Pivaric tracking cells explicitly pass this image size.
            pivaric_imgsz = 1280
            pose_model_m
            """
        )
        model_cell["execution_count"] = None
        model_cell["outputs"] = []
        repaired.append("e0bb2d9c2245eca7")
    elif model_source != (
        "# The following Pivaric tracking cells explicitly pass this image size.\n"
        "pivaric_imgsz = 1280\n"
        "pose_model_m"
    ):
        raise RuntimeError(
            "Legacy model-display cell no longer matches the expected safe rewrite"
        )

    return repaired


def _final_cells() -> list[dict[str, Any]]:
    return [
        _markdown(
            "vision2-steps-title",
            """
            # Vision II — Step Counting (Final Reproducible Pipeline)

            本节是步数确认的最终、可复现入口。前面的单人探索单元保留作开发记录；正式结果统一由
            `src.step_pipeline` 和 `scripts/run_step_counting.py` 产生。

            批处理严格读取 `configs/penalty_slicing_manifest.yaml`，只处理其中 39 个 canonical clips，
            不会把目录里的旧文件名或重复片段算作额外点球。
            """,
        ),
        _markdown(
            "vision2-steps-definition",
            """
            ## 计步定义与人工复核规则

            一步定义为：**有意助跑开始后，一只脚从摆动转为一次新的落地/支撑**。

            - 助跑开始时已经着地的脚不计为新的一步。
            - 射门前最后一次非踢球脚（支撑脚）落地计一步。
            - 踢球脚触球本身不计步，射门后的恢复落地也不计步。
            - `review_required=True`、`status != "ok"` 或低覆盖率/低置信度的结果必须人工复核；
              流程不会为了得到一个数字而强制至少计 1 步。

            `step_frames` 是每次落地对应的证据帧，`step_feet` 给出左右脚顺序；详细逐帧证据保存在
            `outputs/step_diagnostics.json`。
            """,
        ),
        _code(
            "vision2-steps-setup",
            """
            from pathlib import Path
            import csv
            import json
            import subprocess
            import sys

            import yaml
            from IPython.display import display
            from ultralytics import YOLO

            # Locate the repository even if Jupyter was launched from a subdirectory.
            _start = Path.cwd().resolve()
            PROJECT_ROOT = next(
                (
                    candidate
                    for candidate in (_start, *_start.parents)
                    if (candidate / "src" / "step_pipeline.py").is_file()
                ),
                None,
            )
            if PROJECT_ROOT is None:
                raise FileNotFoundError(
                    "Could not find the project root containing src/step_pipeline.py"
                )
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.step_pipeline import process_video_steps

            MANIFEST_PATH = PROJECT_ROOT / "configs" / "penalty_slicing_manifest.yaml"
            CLIPS_DIR = PROJECT_ROOT / "data" / "raw_clips"
            OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "penalty_step_counts.csv"
            DIAGNOSTICS_JSON = PROJECT_ROOT / "outputs" / "step_diagnostics.json"
            RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_step_counting.py"
            MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "yolo11n_pose.pt"

            with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
                manifest_data = yaml.safe_load(handle)

            canonical_clip_ids = [
                f"{match_key}_p{int(penalty['number']):02d}_{penalty['kicker']}"
                for match_key, match in manifest_data.items()
                for penalty in match["penalties"]
            ]
            assert len(canonical_clip_ids) == 39, (
                f"Expected 39 canonical clips, found {len(canonical_clip_ids)}"
            )
            assert len(canonical_clip_ids) == len(set(canonical_clip_ids)), (
                "Canonical clip IDs must be unique"
            )

            missing_clips = [
                clip_id
                for clip_id in canonical_clip_ids
                if not (CLIPS_DIR / f"{clip_id}.mp4").is_file()
            ]
            print(f"Canonical penalties in manifest: {len(canonical_clip_ids)}")
            print(f"Missing canonical clips: {len(missing_clips)}")
            if missing_clips:
                display(missing_clips)
            """,
        ),
        _markdown(
            "vision2-steps-single-note",
            """
            ## 1. 先验证一个视频

            下面用 manifest 的第一条（Falcao）完成一次端到端调用。模型只加载一次；返回字典包含踢球者轨迹、
            kick frame、踢球脚、步数、落地帧、置信度和 QA 标记。确认这个结果能正常产生后再运行 39 人批处理。
            """,
        ),
        _code(
            "vision2-steps-single-run",
            """
            step_pose_model = YOLO(str(MODEL_PATH))

            example_clip_id = canonical_clip_ids[0]
            example_video = CLIPS_DIR / f"{example_clip_id}.mp4"
            example_result = process_video_steps(
                video_path=str(example_video),
                model=step_pose_model,
                clip_id=example_clip_id,
                roi_mode="auto",
                pose_conf=0.10,
                predict_kwargs={"imgsz": 640, "conf": 0.10},
            )

            example_fields = [
                "clip_id",
                "status",
                "runup_start_frame",
                "kick_frame",
                "kicking_foot",
                "step_count",
                "step_frames",
                "step_feet",
                "review_required",
                "auto_accepted",
                "qa_warnings",
            ]
            display({field: example_result.get(field) for field in example_fields})
            """,
        ),
        _markdown(
            "vision2-steps-batch-note",
            """
            ## 2. 按 manifest 运行全部 39 个 canonical clips

            这是正式批处理单元，运行时间会明显长于单视频验证。CLI 内部只加载一次 pose model，并调用
            `process_manifest_steps`；即使个别片段失败，也会为每个 manifest 条目保留一行并标记复核。

            退出码 `0` 表示没有处理失败；退出码 `1` 表示至少一条失败，但 CSV 和 diagnostics 仍会写出，
            应继续运行下面的复核单元定位问题。
            """,
        ),
        _code(
            "vision2-steps-batch-run",
            """
            batch_command = [
                sys.executable,
                str(RUNNER_PATH),
                "--model",
                str(MODEL_PATH),
                "--imgsz",
                "640",
                "--conf",
                "0.10",
                "--manifest",
                str(MANIFEST_PATH),
                "--clips-dir",
                str(CLIPS_DIR),
                "--output-csv",
                str(OUTPUT_CSV),
                "--diagnostics-json",
                str(DIAGNOSTICS_JSON),
            ]

            print("Processing exactly 39 canonical penalties from the manifest...")
            batch_process = subprocess.run(batch_command, cwd=str(PROJECT_ROOT), check=False)
            print(f"Batch exit code: {batch_process.returncode}")
            print(f"CSV: {OUTPUT_CSV}")
            print(f"Diagnostics: {DIAGNOSTICS_JSON}")
            """,
        ),
        _markdown(
            "vision2-steps-results-note",
            """
            ## 3. 读取结果并列出需要确认的球员

            首张表展示全部 39 行的核心字段。第二张表只保留 `review_required=True` 或处理失败的球员；
            这些条目应结合 `step_frames`、`step_feet`、`qa_warnings` 和 diagnostics 中的姿态证据人工确认。
            """,
        ),
        _code(
            "vision2-steps-results",
            """
            if not OUTPUT_CSV.is_file():
                raise FileNotFoundError("Run the 39-clip batch cell before reading results")

            with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
                step_rows = list(csv.DictReader(handle))

            assert len(step_rows) == 39, f"Expected 39 CSV rows, found {len(step_rows)}"

            result_columns = [
                "clip_id",
                "kicker",
                "team",
                "status",
                "kicking_foot",
                "step_count",
                "step_frames",
                "step_feet",
                "confidence",
                "review_required",
                "qa_warnings",
            ]

            def _is_true(value):
                return str(value).strip().lower() == "true"

            try:
                import pandas as pd
            except ImportError:
                print("pandas is unavailable; displaying plain Python rows.")
                display([{key: row.get(key) for key in result_columns} for row in step_rows])
            else:
                display(pd.DataFrame(step_rows)[result_columns])

            review_rows = [
                row
                for row in step_rows
                if _is_true(row.get("review_required"))
                or str(row.get("status", "")).strip().lower() != "ok"
            ]
            review_columns = [
                "clip_id",
                "kicker",
                "status",
                "step_count",
                "step_frames",
                "step_feet",
                "confidence",
                "qa_warnings",
                "error",
            ]

            print(f"Auto-accepted: {len(step_rows) - len(review_rows)} / {len(step_rows)}")
            print(f"Manual review required: {len(review_rows)} / {len(step_rows)}")
            if review_rows:
                if "pd" in globals():
                    display(pd.DataFrame(review_rows)[review_columns])
                else:
                    display([{key: row.get(key) for key in review_columns} for row in review_rows])
            else:
                print("No rows are currently flagged for manual review.")
            """,
        ),
        _code(
            "vision2-steps-diagnostics",
            """
            if not DIAGNOSTICS_JSON.is_file():
                raise FileNotFoundError("Diagnostics JSON was not produced by the batch run")

            with DIAGNOSTICS_JSON.open("r", encoding="utf-8") as handle:
                step_diagnostics = json.load(handle)

            diagnostics_by_clip = {
                item["clip_id"]: item
                for item in step_diagnostics
            }
            review_evidence = [
                {
                    "clip_id": row["clip_id"],
                    "step_count": diagnostics_by_clip[row["clip_id"]].get("step_count"),
                    "step_frames": diagnostics_by_clip[row["clip_id"]].get("step_frames"),
                    "step_feet": diagnostics_by_clip[row["clip_id"]].get("step_feet"),
                    "qa_warnings": diagnostics_by_clip[row["clip_id"]].get("qa_warnings"),
                }
                for row in review_rows
                if row["clip_id"] in diagnostics_by_clip
            ]
            display(review_evidence)
            """,
        ),
        _markdown(
            "vision2-steps-finish",
            """
            ## 完成标准

            当 CSV 正好有 39 行、每行都有 `step_count`，并且复核清单中的异常都已人工确认后，
            才将“每名球员的步数”视为完成。最终提交应同时保留：

            1. `data/processed/penalty_step_counts.csv`（分析表）；
            2. `outputs/step_diagnostics.json`（可复现证据与 QA）；
            3. 本 notebook 和 `src/`、`scripts/` 中的实现代码。
            """,
        ),
    ]


def _confirmed_cells() -> list[dict[str, Any]]:
    return [
        _markdown(
            CONFIRMED_MARKER_ID,
            """
            ## 4. 标注来源纠正：39条AI候选步数，尚无人类复核

            2026-10-08审计发现，旧文件由助手生成，并非用户人工标注。
            `candidate_step_count` 保留AI估计；`confirmed_step_count` 仅在真人复核记录完整时填写。
            当前不能使用候选值作为独立真值计算准确率、MAE或RMSE。历史相关结论已撤回。
            """,
        ),
        _code(
            "vision2-steps-confirmed-merge",
            """
            MANUAL_REVIEWS_CSV = (
                PROJECT_ROOT / "data" / "annotations" / "penalty_step_manual_reviews.csv"
            )
            CONFIRMED_CSV = (
                PROJECT_ROOT / "data" / "processed" / "penalty_step_counts_confirmed.csv"
            )
            MERGE_REVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "merge_manual_step_reviews.py"

            if not MANUAL_REVIEWS_CSV.is_file():
                raise FileNotFoundError(
                    "The 39-row frame-review annotation file is missing: "
                    f"{MANUAL_REVIEWS_CSV}"
                )

            merge_process = subprocess.run(
                [
                    sys.executable,
                    str(MERGE_REVIEW_SCRIPT),
                    "--manifest",
                    str(MANIFEST_PATH),
                    "--clips-dir",
                    str(CLIPS_DIR),
                    "--auto-csv",
                    str(OUTPUT_CSV),
                    "--review-csv",
                    str(MANUAL_REVIEWS_CSV),
                    "--output-csv",
                    str(CONFIRMED_CSV),
                ],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            print(f"Confirmed CSV: {CONFIRMED_CSV}")
            """,
        ),
        _code(
            "vision2-steps-confirmed-results",
            """
            with CONFIRMED_CSV.open("r", encoding="utf-8", newline="") as handle:
                confirmed_rows = list(csv.DictReader(handle))

            assert len(confirmed_rows) == 39, (
                f"Expected 39 confirmed rows, found {len(confirmed_rows)}"
            )
            verified_rows = [row for row in confirmed_rows if row.get("ground_truth_eligible", "").lower() == "true"]
            print(f"Human-verified references: {len(verified_rows)} / {len(confirmed_rows)}")
            print("Candidate estimates are NOT ground truth. Accuracy is not evaluated without verified references.")

            confirmed_columns = [
                "clip_id",
                "kicker",
                "team",
                "confirmed_step_count",
                "candidate_step_count",
                "annotator_type",
                "human_verified",
                "confirmed_landing_frames",
                "manual_confidence",
                "auto_step_count",
                "auto_count_matches_manual",
                "auto_review_required",
                "manual_notes",
            ]

            if "pd" in globals():
                confirmed_df = pd.DataFrame(confirmed_rows)
                display(confirmed_df[confirmed_columns])
                mismatch_df = confirmed_df[
                    confirmed_df["auto_count_matches_manual"].str.lower() != "true"
                ]
                print("Unverified rows and disagreements below require review; this is not an accuracy score.")
                display(mismatch_df[confirmed_columns])
            else:
                display([{key: row.get(key) for key in confirmed_columns} for row in confirmed_rows])
            """,
        ),
        _markdown(
            "vision2-steps-confirmed-finish",
            """
            **使用边界：** 当前 `confirmed_step_count` 应为空；AI候选与自动结果只用于探索。
            只有实际真人复核后填写 reviewer_id、reviewed_at、annotator_type=human、human_verified=true，
            才能生成确认值。不能由程序自动把这些字段改为真人已确认。
            """,
        ),
    ]


def main() -> None:
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    if not BACKUP_PATH.exists():
        shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
        backup_status = "created"
    else:
        backup_status = "kept existing"

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("vision2_penalty_analysis.ipynb is not a supported nbformat 4 notebook")

    repaired = _repair_legacy_cells(notebook)
    section_exists = any(
        SECTION_TAG in cell.get("metadata", {}).get("tags", [])
        for cell in notebook["cells"]
    )
    if not section_exists:
        notebook["cells"].extend(_final_cells())
    confirmed_section_exists = any(
        cell.get("id") == CONFIRMED_MARKER_ID
        for cell in notebook["cells"]
    )
    provenance_repaired = False
    if confirmed_section_exists:
        replacements = {cell["id"]: cell for cell in _confirmed_cells()}
        for index, cell in enumerate(notebook["cells"]):
            fresh = replacements.get(cell.get("id"))
            if fresh is not None and cell.get("source") != fresh.get("source"):
                notebook["cells"][index] = fresh
                provenance_repaired = True
    if not confirmed_section_exists:
        notebook["cells"].extend(_confirmed_cells())

    if repaired or not section_exists or not confirmed_section_exists or provenance_repaired:
        temp_path = NOTEBOOK_PATH.with_name(NOTEBOOK_PATH.name + ".tmp")
        write_mode = "atomic replace"
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    notebook,
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
            try:
                os.replace(temp_path, NOTEBOOK_PATH)
            except PermissionError:
                # Some Windows editors open notebooks without delete-sharing,
                # which blocks atomic replacement but still permits a normal
                # overwrite.  A one-time backup already exists, and the fully
                # serialized temporary file is complete before this fallback.
                with temp_path.open("rb") as source, NOTEBOOK_PATH.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                write_mode = "verified in-place fallback"
        finally:
            if temp_path.exists():
                temp_path.unlink()
    else:
        write_mode = "unchanged"

    print(f"Backup: {backup_status}: {BACKUP_PATH.name}")
    print(f"Legacy cells repaired: {len(repaired)}")
    print(f"Final section appended: {not section_exists}")
    print(f"Confirmed-results section appended: {not confirmed_section_exists}")
    print(f"Notebook write mode: {write_mode}")
    print(f"Notebook cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()
