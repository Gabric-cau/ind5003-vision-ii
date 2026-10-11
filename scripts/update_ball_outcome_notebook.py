"""Append the reproducible Vision II Ball and Outcome workflow to the project notebook."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "vision2_penalty_analysis.ipynb"
BACKUP = ROOT / "notebooks" / "archive" / "vision2_penalty_analysis_before_ball_outcome.ipynb"
MARKER = "VISION_II_BALL_OUTCOME_PIPELINE_V1"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.strip().splitlines()]}


def code(source):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def main():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if any(MARKER in "".join(cell.get("source", [])) for cell in notebook.get("cells", [])):
        print("Ball/Outcome section already present; no changes made.")
        return 0
    if not BACKUP.exists():
        shutil.copy2(NOTEBOOK, BACKUP)
    notebook["cells"].extend(
        [
            markdown(
                f"""
## Vision II — Ball trajectory, final position, and outcome

<!-- {MARKER} -->

本节输出每次点球的球轨迹、画面中的最终位置，以及 `goal` / `no_goal`。
球轨迹使用人工确认的触球帧、YOLO sports-ball 候选、触球前移动球员的脚部锚点、
白球/绿地外观分数和朝球门方向的连续运动约束。像素速度只用于同一视角内比较，
另保留归一化最终位置。Outcome 来自独立的官方赛果标注，不用球轨迹去猜扑救或横梁。
低覆盖或降级搜索的轨迹保留 `ball_review_required=True`。
"""
            ),
            code(
                """
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
BALL_OUTCOME_CSV = PROJECT_ROOT / "data" / "processed" / "penalty_ball_outcome_features.csv"
BALL_OUTCOME_DIAGNOSTICS = PROJECT_ROOT / "outputs" / "ball_outcome_diagnostics.json"
BALL_OUTCOME_LABELS = PROJECT_ROOT / "data" / "annotations" / "penalty_outcomes.csv"

ball_outcome_features = pd.read_csv(BALL_OUTCOME_CSV)
assert len(ball_outcome_features) == 39
display(ball_outcome_features.head())
"""
            ),
            markdown(
                """
### Re-run the scripted extraction

The detector is loaded once and shared across all 39 canonical clips. The command writes a
compact feature CSV and rich JSON containing every candidate and selected trajectory point.
"""
            ),
            code(
                """
%run scripts/run_ball_outcome_analysis.py
"""
            ),
            code(
                """
summary = {
    "rows": len(ball_outcome_features),
    "goals": int((ball_outcome_features["outcome"] == "goal").sum()),
    "no_goals": int((ball_outcome_features["outcome"] == "no_goal").sum()),
    "ball_auto_accepted": int((ball_outcome_features["ball_track_status"] == "auto_accepted").sum()),
    "ball_review_required": int(ball_outcome_features["ball_review_required"].sum()),
}
summary
"""
            ),
            markdown(
                """
### QA evidence

Run `%run scripts/render_ball_qa.py` to regenerate the six all-clip contact sheets, or add
`--review-only` to render only conservative review flags. Yellow circles mark selected ball
detections and magenta crosses mark eligible contact-foot anchors.
"""
            ),
        ]
    )
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Appended Ball/Outcome workflow; cells={len(notebook['cells'])}")
    print(f"Backup: {BACKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
