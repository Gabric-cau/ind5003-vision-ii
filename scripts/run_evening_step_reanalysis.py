"""Versioned 39-clip step/foot reanalysis with explicit contact-source hierarchy."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
cv2.setNumThreads(2)
from scripts.run_penalty_phase_experiment import process,load,save,digest
from scripts.run_terminal_footwork import analyse
from scripts.validate_terminal_footwork import check_result
from src.style_aware_steps import analyze_style_aware_steps
from src.step_pipeline import _track_pose_sequence
from src.kicker_analysis import KickerFeatureExtractor
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
UP=RUN/"phase_v4";OUT=RUN/"steps_v4/classified"
for d in (UP/"classified",UP/"contact_probe",UP/"shot_rescue/poses",UP/"shot_rescue/contacts",OUT):d.mkdir(parents=True,exist_ok=True)
baseline={r["clip_id"]:r for r in load(ROOT/"outputs/style_aware_steps_2026-10-09/classified/summary.json")["rows"]}
rows=[]
for cid in baseline:
    dest=OUT/(cid+".json")
    if dest.exists():raise FileExistsError(dest)
    cp=RUN/"contact_roi"/(cid+".json")
    if not cp.exists():cp=RUN/"contact_transition"/(cid+".json")
    posepath=ROOT/"outputs/shot_repair_2026-10-09/all_clips_final"/(cid+".json")
    override=ROOT/"outputs/phase_optimization_2026-10-09/shot_rescue/poses"/(cid+".json")
    kind="historical_later_shot_rescue" if override.exists() else None
    if cid=="rus_cro_p02_brozovic_cro11":
        cp=RUN/"medium_pose_causal"/(cid+"_contact.json")
        override=RUN/"medium_pose_causal"/(cid+"_pose.json")
        kind="same_shot_medium_pose_tiles_and_causal_first_departure"
    shutil.copy2(cp,UP/"contact_probe"/(cid+".json"))
    if override.exists():
        shutil.copy2(override,UP/"shot_rescue/poses"/(cid+".json"))
        shutil.copy2(cp,UP/"shot_rescue/contacts"/(cid+".json"))
    args=SimpleNamespace(contact_dir=UP/"contact_probe",output_dir=UP/"classified",rescue_dir=UP/"shot_rescue",
                         refined_dir=None,recompute_camera=False)
    r=process(posepath,args)
    r["experimental_pose_override_kind"]=kind
    if kind and "medium_pose" in kind:r["pose_source"]="experimental_medium_pose_tiles_same_shot"
    contact=load(cp);pose=load(override if override.exists() else posepath)
    track=next((t for t in pose["tracks"] if t["track_id"]==contact.get("contact_associated_track_id")),{"observations":[]})
    seq=_track_pose_sequence(track,.35)
    foot={"kicking_foot":"unknown"}
    if contact.get("contact_frame") is not None and len(seq["frames"])>=2:
        foot=KickerFeatureExtractor().classify_kicking_foot_from_motion(
            **{k:seq[k] for k in ("frames","left_ankles","right_ankles","left_hips","right_hips")},
            kick_frame=contact["contact_frame"],window=4)
    r["foot_analysis"]={"result":foot,"method":"hip_relative_ankle_motion_on_contact_associated_actor",
                        "keypoint_confidence_gate":.35,"human_verified":False,"accuracy_validated":False}
    save(UP/"classified"/(cid+".json"),r)
    r=analyse(UP/"classified"/(cid+".json"),UP)
    r["step_analysis"]=analyze_style_aware_steps(r)
    r["schema_version"]="evening-v4-experimental-no-human-verification"
    errors=check_result(r)
    if errors:raise ValueError((cid,errors))
    r["experiment_inputs"]={"contact_candidate_path":str(cp),"contact_candidate_sha256":digest(cp),
            "medium_pose_candidate_visual_review":"AI_only_not_ground_truth" if kind and "medium" in kind else None}
    save(dest,r)
    row={"clip_id":cid,"baseline_count":baseline[cid]["candidate_step_count"],
         "candidate_step_count":r["step_analysis"]["candidate_step_count"],"attempt_style":r["step_analysis"]["attempt_style"],
         "contact_status":contact["status"],"contact_frame":contact.get("contact_frame"),
         "kicking_foot":foot.get("kicking_foot"),"phase_status":r["phase_result"]["status"],
         "override_kind":kind,"human_verified":False}
    rows.append(row);save(OUT/"summary.json",{"rows":rows,"clips":len(rows),"count_available":sum(v["candidate_step_count"] is not None for v in rows),
        "attempt_styles":dict(Counter(v["attempt_style"] for v in rows)),"human_verified":False,"accuracy_validated":False})
    print(json.dumps(row),flush=True)
