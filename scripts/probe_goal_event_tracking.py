"""Reanalyse dense ball detections with optional camera-compensated white motion blobs."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from src.event_ball_tracking import track_ball_event,summarize_event_cues
from src.goal_event_tracker_v2 import track_goal_event,track_event_segments,summarize_segment_cues
from src.goal_plane_rectification import project_image_point
from src.keeper_kit_evidence import learn_kit_profile,annotate_kit_ambiguity
cv2.setNumThreads(2)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def motion_blobs(cache):
    cap=cv2.VideoCapture(cache["video_path"]);cap.set(1,cache["start_frame"])
    previous=None;previous_tf=None;extra={}
    for fid in range(cache["start_frame"],cache["end_frame_exclusive"]):
        ok,frame=cap.read()
        if not ok:raise ValueError("Input decode failed")
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);tf=cache["camera"]["transforms"].get(str(fid));extra[str(fid)]=[]
        if tf is not None and previous_tf is not None:
            relative=np.linalg.inv(np.array(tf))@np.array(previous_tf)
            warped=cv2.warpPerspective(previous,relative,(cache["width"],cache["height"]))
            diff=cv2.absdiff(gray,warped);hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask=((hsv[...,1]<105)&(hsv[...,2]>145)&(diff>18)).astype(np.uint8)*255
            goal=cache["goal_proposals"][str(fid)].get("selected")
            if goal:
                mask[:max(0,int(min(c[1] for c in goal["corners"])-20))]=0
            mask[int(.85*cache["height"]):]=0
            n,lab,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
            for i in range(1,n):
                x,y,w,h,area=stats[i]
                if not 12<=area<=1600 or not 3<=w<=110 or not 3<=h<=45 or max(w/h,h/w)>8:continue
                cx,cy=centroids[i];patch=diff[y:y+h,x:x+w]
                extra[str(fid)].append({"frame":fid,"x":float(cx),"y":float(cy),"width":float(w),"height":float(h),
                    "confidence":0.,"sources":["motion_blob"],"detection_id":f"{fid}:blob:{i}",
                    "stabilized_xy":project_image_point(tf,[cx,cy]),"motion_mean_abs_gray":float(np.mean(patch)),
                    "blob_area":int(area)})
        previous=gray;previous_tf=tf
    cap.release();return extra


def source_time_indices(cache):
    """Recognize near-identical encoded frames; keep original frame IDs for audit."""
    cap=cv2.VideoCapture(cache["video_path"]);cap.set(1,cache["start_frame"])
    previous=None;index=0;result={}
    for fid in range(cache["start_frame"],cache["end_frame_exclusive"]):
        ok,frame=cap.read()
        if not ok:raise ValueError("Incomplete source decode")
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mad=float(np.mean(cv2.absdiff(gray,previous))) if previous is not None else None
        duplicate=mad is not None and mad<.35
        if not duplicate:index+=1
        result[fid]={"motion_time":index,"duplicate_frame":duplicate,"source_frame_mad":mad}
        previous=gray
    cap.release();return result


def add_kit_evidence(cache,people,candidates):
    # The prior contributes only a pre-kick location, never an outcome label.
    p=ROOT/'outputs/optimization_evening_2026-10-09/goal_appearance_v2'/(cache['clip_id']+'.json')
    prior=load(p).get('selected_candidate') or {};pre=prior.get('pre_trajectory',[])
    if not pre:return {'status':'unavailable_prior','frames':{}}
    row=min(pre,key=lambda r:abs(r[0]*cache['fps']-(cache['kick_frame']-8)))
    fid=round(row[0]*cache['fps']);center=np.array(row[1:3])
    boxes=[q for q in people.get(str(fid),[]) if q['confidence']>=.25
           and np.linalg.norm((np.array(q['bbox'][:2])+q['bbox'][2:])/2-center)<60]
    if not boxes:return {'status':'unavailable_seed_box','frames':{}}
    box=min(boxes,key=lambda q:np.linalg.norm((np.array(q['bbox'][:2])+q['bbox'][2:])/2-center))['bbox']
    cap=cv2.VideoCapture(cache['video_path']);cap.set(1,fid);ok,frame=cap.read()
    if not ok:raise ValueError('Missing kit seed frame')
    profile=learn_kit_profile(frame,box)
    if profile is None:cap.release();return {'status':'unavailable_profile','frames':{}}
    cap.set(1,cache['start_frame']);observations={}
    for f in range(cache['start_frame'],cache['end_frame_exclusive']):
        ok,frame=cap.read()
        if not ok:raise ValueError('Kit frame decode failed')
        g=cache['goal_proposals'][str(f)].get('selected')
        if not g:continue
        x0,y0,x1,y1=g['bbox'];w=x1-x0;h=y1-y0
        roi=[max(0,x0-.4*w),y0+.3*h,min(cache['width'],x1+.8*w),min(cache['height'],y1+.6*h)]
        evidence=annotate_kit_ambiguity(frame,profile,roi,candidates[str(f)],center)
        if evidence:observations[str(f)]=evidence;center=np.array(evidence['center'])
    cap.release()
    return {'status':'candidate','profile':profile,'seed_frame':fid,'seed_bbox':box,
            'frames':observations,'human_verified':False,'source_path':str(p)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--cache-dir",type=Path,default=ROOT/"outputs/goal_event_evidence_2026-10-10/detections")
    parser.add_argument("--directory",default="tracking_v1");parser.add_argument("--v2",action="store_true");args=parser.parse_args()
    out=args.cache_dir.parent/args.directory;out.mkdir(parents=True,exist_ok=True);rows=[]
    for file in sorted(args.cache_dir.glob("*.json")):
        cache=load(file)
        if cache.get("status")!="cached_candidates":continue
        dest=out/file.name
        if dest.exists():raise FileExistsError(dest)
        cid=cache["clip_id"];pose=load(Path(cache["source_paths"]["pose"]));contact=cache["contact"]
        tid=contact["contact_associated_track_id"];track=next(t for t in pose["tracks"] if t["track_id"]==tid)
        anchors=[]
        for obs in track["observations"]:
            tf=cache["camera"]["transforms"].get(str(obs["frame"]))
            if abs(obs["frame"]-cache["kick_frame"])<=3 and tf is not None:
                for j in (15,16):
                    if obs["keypoint_confidences"][j]>=.25:
                        p=project_image_point(tf,obs["keypoints"][j])
                        if p is not None:anchors.append(p)
        fid=min((int(f) for f,g in cache["goal_proposals"].items() if g.get("selected")),key=lambda f:abs(f-cache["kick_frame"]))
        goal=cache["goal_proposals"][str(fid)]["selected"];tf=cache["camera"]["transforms"][str(fid)]
        center=project_image_point(tf,np.mean(goal["corners"],axis=0));width=goal["bbox"][2]-goal["bbox"][0]
        blobs=motion_blobs(cache);variants={};times=source_time_indices(cache)
        person_path=file.with_name(cid+"_people.json")
        people=load(person_path)["frames"] if person_path.exists() else {}
        kit_evidence=None
        for mode,use_blobs,allow_turn in [("detector_straight",False,False),("detector_turn",False,True),
                                         ("detector_motion_straight",True,False),("detector_motion_turn",True,True)]:
            candidates={f:list(v)+(blobs.get(f,[]) if use_blobs else []) for f,v in cache["detections"].items()}
            candidates={f:[{**d,**times[int(f)]} for d in ds] for f,ds in candidates.items()}
            for f,ds in candidates.items():
                for d in ds:
                    d["person_overlap"]=any(p["confidence"]>=.25 and p["bbox"][0]-3<=d["x"]<=p["bbox"][2]+3
                        and p["bbox"][1]-3<=d["y"]<=p["bbox"][3]+3 for p in people.get(f,[]))
            if args.v2:kit_evidence=add_kit_evidence(cache,people,candidates)
            if args.v2 and mode=='detector_motion_turn':
                trajectory,diag=track_event_segments(candidates,cache["kick_frame"],anchors,center,goal_width=width)
            elif args.v2:
                trajectory,diag=track_goal_event(candidates,cache["kick_frame"],anchors,center,allow_turn=allow_turn,goal_width=width)
            else:
                trajectory,diag=track_ball_event(candidates,cache["kick_frame"],anchors,center,allow_turn=allow_turn)
            cues=summarize_event_cues(trajectory,center,width)
            if args.v2 and mode=='detector_motion_turn':
                cues=summarize_segment_cues(trajectory,diag,(kit_evidence or {}).get('frames'))
            variants[mode]={"trajectory":trajectory,"tracking":diag,"events":cues}
        record={"clip_id":cid,"cache_path":str(file),"video_path":cache["video_path"],"anchors":anchors,
                "goal_reference_center":center,"goal_reference_width":width,"variants":variants,
                "human_verified":False,"accuracy_validated":False,"outcome_labels_read":False,
                "duplicate_frames":[f for f,t in times.items() if t["duplicate_frame"]],"keeper_kit_evidence":kit_evidence,
                "code_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),ROOT/'src/goal_event_tracker_v2.py',ROOT/'src/keeper_kit_evidence.py',ROOT/'src/event_ball_tracking.py']}}
        save(dest,record)
        row={"clip_id":cid,"variants":{k:{"n":len(v["trajectory"]),"start":v["trajectory"][0]["frame"] if v["trajectory"] else None,
             "end":v["trajectory"][-1]["frame"] if v["trajectory"] else None,"events":v["events"],"tracking":v["tracking"]} for k,v in variants.items()}}
        rows.append(row);print(json.dumps(row),flush=True)
    save(out/"summary.json",{"rows":rows,"clips":len(rows),"accuracy_validated":False,"human_verified":False})


if __name__=="__main__":main()
