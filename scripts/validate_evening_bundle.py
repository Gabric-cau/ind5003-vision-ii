"""Evidence-bundle integrity checks; no labels or model accuracy inferred."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
def load(p):return json.loads(p.read_text(encoding="utf-8"))
errors=[]
def check(ok,message):
    if not ok:errors.append(message)
joined=load(RUN/"integration/canonical_results_39.json")
rows=joined["rows"];check(len(rows)==39 and len({r["clip_id"] for r in rows})==39,"canonical IDs")
for row in rows:
    cid=row["clip_id"]
    check(not row["human_verified"] and not row["accuracy_validated"],cid+" provenance")
    check(row["steps"]["verified_count"] is None and row["foot"]["verified_foot"] is None,cid+" truth fields")
    check(row["outcome"]["automatic_prediction"]=="unknown" and not row["outcome"]["used_by_predictor"],cid+" outcome leakage flag")
    check(row["ball"]["event_final_position"] is None,cid+" endpoint semantics")
    for section in ("steps","foot","goalkeeper","camera"):
        check(Path(row[section]["source"]).is_file(),cid+" missing "+section)
    gk=row["goalkeeper"]
    if not gk["observability"]["window_supported"]:
        check(gk["window_guarded_direction"]=="unknown" and not gk["guarded_qa_accepted"],cid+" censored dive promoted")
    # File schemas contain either a list or a named list of camera segments.
    camera=load(Path(row["camera"]["source"]))
    segments=camera["segments"]
    check(segments[0]["start_frame"]==0,cid+" camera start")
    check(all(a["end_frame_exclusive"]==b["start_frame"] for a,b in zip(segments,segments[1:])),cid+" camera gaps")
    check(all(s["verified_camera_class"]=="unknown" and s["slow_motion_status"]=="unknown" for s in segments),cid+" unsupported camera assertion")
metrics=load(RUN/"integration/metrics.json")
check(sum(metrics["attempt_styles"].values())==39,"style denominator")
check(sum(metrics["candidate_step_count_distribution"].values())==39,"step denominator")
check(sum(metrics["paired_count_change_distribution"].values())==31,"paired denominator")
check(Counter(r["outcome"]["external_label"] for r in rows)=={"goal":26,"no_goal":13},"external labels")
check(metrics["tests"]["passed"] and metrics["tests"]["total_tests"]==168,"test evidence")
snapshot=load(RUN/"integration/source_snapshot.json");path=Path(snapshot["path"])
check(hashlib.sha256(path.read_bytes()).hexdigest()==snapshot["sha256"],"snapshot hash")
with zipfile.ZipFile(path) as z:check(z.testzip() is None,"snapshot CRC")
report={"passed":not errors,"errors":errors,"canonical_rows":len(rows),
        "ground_truth_inferred":False,"accuracy_validated":False,"scope":"Join, provenance, observation semantics, intervals and source archive integrity"}
(RUN/"integration/bundle_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2));sys.exit(0 if not errors else 1)
