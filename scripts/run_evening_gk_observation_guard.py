"""Reject whole-window dive conclusions from truncated tracked prefixes."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.goalkeeper_observation_guard import guard_postkick_observations
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
SOURCE=RUN/"goal_appearance_v2";OUT=RUN/"goal_observation_guard"
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for path in sorted(SOURCE.glob("*.json")):
    if path.name=="summary.json":continue
    original=json.loads(path.read_text(encoding="utf-8"))
    if "kick_frame" not in original:
        step=json.loads((RUN/"steps_v4/classified"/path.name).read_text(encoding="utf-8"))
        original.update(kick_frame=step["contact"]["contact_frame"],fps=step["fps"],
                        video_path=step["video_path"],kick_frame_source="independent_contact_candidate")
    # Historical runner keeps its original whole-window semantics. For the
    # current separate-direction policy use run_goalkeeper_direction_update.py.
    out=guard_postkick_observations(original,policy='legacy_full_window_v1')
    out["source_candidate_path"]=str(path)
    out["observation_guard_sha256"]=hashlib.sha256((ROOT/"src/goalkeeper_observation_guard.py").read_bytes()).hexdigest()
    dest=OUT/path.name
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8"))!=out:raise FileExistsError(dest)
    else:dest.write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
    rows.append({"clip_id":out["clip_id"],"pre_guard_auto_accepted":out["pre_guard_auto_accepted"],
                 **{k:out[k] for k in ("auto_accepted","candidate_dive_direction","dive_direction","dive_height","dive_observability")}})
summary={"clips":len(rows),"pre_guard_qa":sum(r["pre_guard_auto_accepted"] for r in rows),
         "guarded_qa":sum(r["auto_accepted"] for r in rows),
         "window_supported":sum(r["dive_observability"]["window_supported"] for r in rows),
         "directions":dict(Counter(r["dive_direction"] for r in rows)),"rows":rows,
         "human_verified":False,"accuracy_validated":False,
         "scope":"Stricter completeness reporting; not an independent identity or accuracy test"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k!="rows"},indent=2))
