"""Join real canonical records with explicit provenance; never the demo table.

Only this descriptive reporting stage reads external outcome labels. Prediction
experiments stay independent of those labels. No candidate becomes ground truth.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv,hashlib,importlib,importlib.metadata,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT/".venv/Lib/site-packages"))
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
OUT=RUN/"integration";OUT.mkdir(parents=True,exist_ok=True)


def load(path):return json.loads(path.read_text(encoding="utf-8"))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def csv_map(name):
    with (ROOT/"data/processed"/name).open(encoding="utf-8",newline="") as f:
        return {r["clip_id"]:r for r in csv.DictReader(f)}


def build():
    manifest=yaml.safe_load((ROOT/"configs/penalty_slicing_manifest.yaml").read_text(encoding="utf-8"))
    meta={f"{key}_p{int(p['number']):02d}_{p['kicker']}":{
        "match_key":key,"match_title":m["match_title"],"kicker":p["kicker"],"team":p.get("team"),
        "penalty_number":int(p["number"])} for key,m in manifest.items() for p in m["penalties"]}
    step_summary=load(RUN/"steps_v4/classified/summary.json")
    old_step=csv_map("penalty_step_counts.csv");old_gk=csv_map("penalty_goalkeeper_features.csv")
    old_ball=csv_map("penalty_ball_outcome_features.csv")
    assert len(meta)==39 and set(meta)==set(old_step)==set(old_gk)==set(old_ball)
    assert set(meta)=={r["clip_id"] for r in step_summary["rows"]}
    rows=[]
    for s in step_summary["rows"]:
        cid=s["clip_id"];name=cid+".json"
        step=load(RUN/"steps_v4/classified"/name)
        foot=load(RUN/"foot_sensitivity"/name)
        gk=load(RUN/"goal_observation_guard"/name)
        ball_path=RUN/"ball_extended_v3"/name
        ball=load(ball_path) if ball_path.exists() else None
        record={"clip_id":cid,**meta[cid],"video_path":step["video_path"],
          "steps":{"baseline_v3_candidate_count":s["baseline_count"],
                   "v4_candidate_count":s["candidate_step_count"],
                   "verified_count":None,"attempt_style":s["attempt_style"],
                   "count_partition":step["step_analysis"]["count_partition"],
                   "parameter_range":step["step_analysis"]["count_sensitivity"]["threshold_and_contact_range"],
                   "partial_approach_known":cid=="rus_cro_p02_brozovic_cro11",
                   "source":str(RUN/"steps_v4/classified"/name)},
          "contact":step["contact"],
          "foot":{"baseline_candidate":old_step[cid]["kicking_foot"],
                  "nominal_v4_candidate":foot["nominal_candidate"],
                  "parameter_stable_candidate":foot["stable_candidate"],
                  "parameter_stable":foot["parameter_stable"],"verified_foot":None,
                  "source":str(RUN/"foot_sensitivity"/name)},
          "goalkeeper":{"baseline_qa_accepted":old_gk[cid]["auto_accepted"].lower()=="true",
                        "experimental_candidate_direction":gk["candidate_dive_direction"],
                        "experimental_candidate_height":gk["candidate_dive_height"],
                        "window_guarded_direction":gk["dive_direction"],
                        "window_guarded_height":gk["dive_height"],
                        "guarded_qa_accepted":gk["auto_accepted"],
                        "pre_guard_qa_accepted":gk["pre_guard_auto_accepted"],
                        "observability":gk["dive_observability"],"verified_identity":False,
                        "source":str(RUN/"goal_observation_guard"/name)},
          "ball":{"baseline_qa_accepted":old_ball[cid]["ball_track_status"]=="auto_accepted",
                  "experiment_cohort_member":ball is not None,
                  "experimental_status":ball.get("status") if ball else "not_retested",
                  "fixed_full32_continuity_qa":((ball.get("variants",{}).get("full_32") or {}).get("continuity_qa_accepted")) if ball else None,
                  "experimental_source":str(ball_path) if ball else None,
                  "event_final_position":None,"speed_units":"image_pixels_per_second_not_meters_per_second"},
          "outcome":{"external_label":old_ball[cid]["outcome"],
                     "external_label_detail":old_ball[cid]["outcome_detail"],
                     "recorded_source":old_ball[cid]["outcome_source"],
                     "source_verified_in_this_experiment":False,
                     "automatic_prediction":"unknown","used_by_predictor":False},
          "homography":{"validated_ground_plane":False},
          "camera":{"source":str(RUN/"camera_canonical"/name),"verified_class":"unknown"},
          "human_verified":False,"accuracy_validated":False}
        assert record["steps"]["v4_candidate_count"] is not None
        assert record["outcome"]["automatic_prediction"]=="unknown"
        rows.append(record)
    assert len({r["clip_id"] for r in rows})==39
    joined={"schema_version":"evening-evidence-join-v1","canonical_count":39,
       "generated_at_utc":datetime.now(timezone.utc).isoformat(),
       "policy":"Candidate coverage, algorithmic QA, parameter stability and real accuracy are distinct.",
       "deprecated_input_not_read":"data/processed/penalties_features.csv (49 demonstration rows)",
       "human_ground_truth_records":0,"rows":rows}
    save(OUT/"canonical_results_39.json",joined)
    old_counts=[r["steps"]["baseline_v3_candidate_count"] for r in rows if r["steps"]["baseline_v3_candidate_count"] is not None]
    paired=[r for r in rows if r["steps"]["baseline_v3_candidate_count"] is not None]
    styles=dict(Counter(r["steps"]["attempt_style"] for r in rows))
    delta=dict(Counter(r["steps"]["v4_candidate_count"]-r["steps"]["baseline_v3_candidate_count"] for r in paired))
    ball_summary=load(RUN/"ball_extended_v3/summary.json")
    ball_counts={v:sum(bool(r.get("variants",{}).get(v,{}).get("continuity_qa")) for r in ball_summary["rows"])
                 for v in ("full_32","full_60","corridor_32","corridor_60")}
    metrics={"canonical_clips":39,"baseline_count_available":len(old_counts),"v4_count_available":39,
       "candidate_step_count_distribution":dict(Counter(r["steps"]["v4_candidate_count"] for r in rows)),
       "attempt_styles":styles,"paired_count_change_distribution":delta,
       "new_count_rows":[{"clip_id":r["clip_id"],"candidate_count":r["steps"]["v4_candidate_count"],
                          "parameter_range":r["steps"]["parameter_range"],"contact_status":r["contact"]["status"],
                          "partial_approach_known":r["steps"]["partial_approach_known"]}
                         for r in rows if r["steps"]["baseline_v3_candidate_count"] is None],
       "foot_parameter_stable":sum(r["foot"]["parameter_stable"] for r in rows),
       "gk_guard":{k:v for k,v in load(RUN/"goal_observation_guard/summary.json").items() if k!="rows"},
       "ball_review_cohort":len(ball_summary["rows"]),"ball_continuity_qa_counts":ball_counts,
       "camera_segments":load(RUN/"camera_canonical/summary.json")["segments"],
       "external_outcome_labels":dict(Counter(r["outcome"]["external_label"] for r in rows)),
       "human_ground_truth_records":0,"accuracy_estimate":None,
       "tests":load(RUN/"validation.json")}
    save(OUT/"metrics.json",metrics)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(2,2,figsize=(11.4,7.1),constrained_layout=True)
    counts=Counter(r["steps"]["v4_candidate_count"] for r in rows)
    ax[0,0].bar(list(sorted(counts)),[counts[k] for k in sorted(counts)],color="#31698a")
    ax[0,0].set(title="Candidate step counts (39 attempts)",xlabel="Visible-approach candidate count",ylabel="Attempts")
    ax[0,0].set_xticks(range(2,max(counts)+1))
    style_values=[styles.get(k,0) for k in ("continuous_approach_candidate","terminal_adjustment_candidate","undetermined")]
    assert sum(style_values)==39
    ax[0,1].bar(["Continuous","Terminal change","Undetermined"],
        style_values,color=["#31698a","#cf8933","#919aa5"])
    ax[0,1].set(title="Attempt-style candidates (39 attempts)",ylabel="Attempts")
    for x,y in enumerate(style_values):ax[0,1].text(x,y+.3,str(y),ha="center")
    x=[r["steps"]["baseline_v3_candidate_count"] for r in paired];y=[r["steps"]["v4_candidate_count"] for r in paired]
    pairs=Counter(zip(x,y));ax[1,0].scatter([p[0] for p in pairs],[p[1] for p in pairs],s=[pairs[p]*28 for p in pairs],color="#31698a",alpha=.7)
    ax[1,0].plot([0,max(x)+1],[0,max(x)+1],"--",color="gray",linewidth=1)
    ax[1,0].set(title="Paired candidates (31 attempts; dot size = repeats)",xlabel="Baseline v3 count",ylabel="Experimental v4 count")
    ax[1,1].bar(["Full / 32f","Full / 60f","ROI / 32f","ROI / 60f"],list(ball_counts.values()),color="#31698a")
    ax[1,1].set(title="Ball continuity QA (13 review-flagged attempts)",ylabel="Rule-screened candidates",ylim=(0,13))
    for x,y in enumerate(ball_counts.values()):ax[1,1].text(x,y+.2,str(y),ha="center")
    fig.suptitle("Real canonical records — descriptive diagnostics, NOT accuracy",fontsize=15)
    fig.savefig(OUT/"actual39_diagnostics.png",dpi=170);plt.close(fig)
    packages={}
    for package,module in [("numpy","numpy"),("scipy","scipy"),("pandas","pandas"),("PyYAML","yaml"),("opencv-python","cv2"),
                           ("ultralytics","ultralytics"),("torch","torch"),("torchvision","torchvision"),("Pillow","PIL"),("matplotlib","matplotlib")]:
        mod=importlib.import_module(module)
        packages[package]={"version":importlib.metadata.version(package),"module_path":str(Path(mod.__file__).resolve())}
    save(OUT/"environment.json",{"python":sys.version,"executable":sys.executable,"packages":packages,
         "project_venv_appended":str(ROOT/".venv/Lib/site-packages"),"clean_install_tested":False,
         "warning":"Hybrid existing environment; Python launcher emits relocated-base warning but selected runs exit successfully."})
    snapshot=OUT/"source_snapshot.zip"
    with zipfile.ZipFile(snapshot,"w",zipfile.ZIP_DEFLATED) as z:
        paths=list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").glob("*.py"))+list((ROOT/"tests").glob("*.py"))
        paths+=[ROOT/"requirements-evening-experiment.txt",ROOT/"configs/penalty_slicing_manifest.yaml"]
        for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    save(OUT/"source_snapshot.json",{"path":str(snapshot),"sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest(),
         "scope":"Current source and tests only, not weights/videos/results or exact historical experiment implementations"})
    print(json.dumps({k:v for k,v in metrics.items() if k not in ("new_count_rows","tests")},indent=2))


if __name__=="__main__":build()
