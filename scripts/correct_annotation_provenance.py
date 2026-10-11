"""One-time, backed-up correction of AI labels formerly described as manual."""

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.merge_manual_step_reviews import merge, DEFAULT_MANIFEST, DEFAULT_CLIPS_DIR, DEFAULT_AUTO_CSV, DEFAULT_OUTPUT


def main():
    backup = ROOT / "outputs/archive/provenance_before_2026-10-08"
    backup.mkdir(parents=True, exist_ok=True)
    names = ["penalty_step_manual_reviews.csv", "penalty_kick_frames.csv", "penalty_goalkeeper_track_reviews.csv"]
    summary = {}
    for name in names:
        path = ROOT / "data/annotations" / name
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames)
            rows = list(reader)
        if not (backup / name).exists():
            shutil.copy2(path, backup / name)
        extra = ["annotator_type", "human_verified", "reviewer_id", "reviewed_at", "provenance_note"]
        for row in rows:
            # Preserve any later, explicitly supplied human annotation.
            if row.get("annotator_type") == "human":
                continue
            row.update(annotator_type="ai_assisted", human_verified="false", reviewer_id="", reviewed_at="",
                       provenance_note="Created by assistant in prior session; no user human review. Audit 2026-10-08.")
            if "source" in row:
                row["source"] = "ai_visual_estimate_unverified"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fields + [key for key in extra if key not in fields])
            writer.writeheader()
            writer.writerows(rows)
        summary[name] = len(rows)
    if DEFAULT_OUTPUT.exists() and not (backup / DEFAULT_OUTPUT.name).exists():
        shutil.copy2(DEFAULT_OUTPUT, backup / DEFAULT_OUTPUT.name)
    merged = merge(DEFAULT_MANIFEST, DEFAULT_CLIPS_DIR, [ROOT / "data/annotations" / names[0]], DEFAULT_AUTO_CSV, DEFAULT_OUTPUT)
    summary["human_verified_step_rows"] = sum(row["ground_truth_eligible"] for row in merged)
    notebook = ROOT / "notebooks/vision2_penalty_analysis.ipynb"
    if not (backup / notebook.name).exists():
        shutil.copy2(notebook, backup / notebook.name)
    from scripts.update_step_counting_notebook import main as update_notebook
    update_notebook()
    content = json.loads(notebook.read_text(encoding="utf-8"))
    notice_id = "vision2-provenance-correction-20261008"
    if not any(cell.get("id") == notice_id for cell in content["cells"]):
        content["cells"].insert(0, {"id": notice_id, "cell_type": "markdown", "metadata": {}, "source": [
            "# 2026-10-08 标注来源更正\n",
            "旧步数、触球帧和守门员种子标签由AI助手生成，并非用户或独立真人标注。当前真人确认步数参考为0条。\n",
            "旧版manual/confirmed命名、历史文本及输出不能作为真人复核证据；基于这些候选的准确率、MAE、RMSE结论撤回。\n",
            "候选值已保留，确认字段为空；详见 docs/vision2_optimization_audit_2026-10-08.md。\n",
            "守门员/足球脚本如需复用未验证输入，须显式添加 --allow-unverified-annotations，并标为探索实验。\n"
        ]})
        notebook.write_text(json.dumps(content, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    # Correct attribution fields in existing derived products, retaining measurements and QA decisions.
    path = ROOT / "data/processed/penalty_goalkeeper_features.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    if not (backup / path.name).exists():
        shutil.copy2(path, backup / path.name)
    for row in rows:
        row["goalkeeper_identity_reviewed"] = "False"
        row["kick_frame_source"] = "ai_visual_estimate_unverified"
        row["input_annotation_status"] = "ai_assisted_unverified"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields + ([] if "input_annotation_status" in fields else ["input_annotation_status"]))
        writer.writeheader()
        writer.writerows(rows)
    summary["accuracy_metrics"] = None
    ball_path = ROOT / "data/processed/penalty_ball_outcome_features.csv"
    with ball_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        ball_rows = list(reader)
    if not (backup / ball_path.name).exists():
        shutil.copy2(ball_path, backup / ball_path.name)
    for row in ball_rows:
        row["input_annotation_status"] = "ai_assisted_unverified"
        row["kick_frame_source"] = "ai_visual_estimate_unverified"
    with ball_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields + [key for key in ("input_annotation_status", "kick_frame_source") if key not in fields])
        writer.writeheader()
        writer.writerows(ball_rows)
    summary["reason"] = "No independently human-verified reference labels; prior agreement/MAE/RMSE claims withdrawn as accuracy."
    (ROOT / "outputs/annotation_provenance_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
