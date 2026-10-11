"""Two bounded, reproducible side-by-side goal-plane video demonstrations.

No generative pixels, no outcome labels, no fabricated trajectory interpolation.
This prototype rectifies visible image pixels; it is not a real front camera.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,subprocess,sys,unittest,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.goal_geometry import detect_goal_frame
from src.goal_plane_rectification import fit_goal_plane,project_image_point,ball_projection_evidence
cv2.setNumThreads(2)
BASE=ROOT/"outputs/optimization_evening_2026-10-09"
TARGET=(60.,85.,660.,285.)
FFMPEG=ROOT/"outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"


def load(p):return json.loads(p.read_text(encoding="utf-8"))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label(im,text,xy,color=(240,240,240),scale=.54):
    cv2.putText(im,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def dot(im,xy,color):
    if xy is not None and -10<xy[0]<im.shape[1]+10 and -10<xy[1]<im.shape[0]+10:
        p=tuple(np.rint(xy).astype(int));cv2.circle(im,p,7,color,2,cv2.LINE_AA)


def run_clip(cid,out):
    step_path=BASE/"steps_v4/classified"/(cid+".json")
    ball_path=BASE/"ball_extended_v3"/(cid+".json")
    gk_path=BASE/"goal_appearance_v2"/(cid+".json")
    step=load(step_path);ball=load(ball_path);gk=load(gk_path)
    video=Path(step["video_path"]);source_hash=sha(video)
    kick=step["contact"]["contact_frame"];fps=float(step["fps"])
    start=max(step["shot_start_frame"],kick-round(.5*fps))
    end=min(step["shot_end_frame_exclusive"],kick+round(2.0*fps)+1)
    bmap={r["frame"]:r for r in ball.get("variants",{}).get("full_32",{}).get("trajectory",[])}
    selected=gk.get("selected_candidate") or {}
    gmap={round(r[0]*fps):r[1:3] for r in selected.get("pre_trajectory",[])+selected.get("post_trajectory",[])}
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    output=out/(cid+"_goal_plane_slow.mp4")
    if output.exists():raise FileExistsError(output)
    # Encode the untouched frame sequence at one third speed for inspection.
    cmd=[str(FFMPEG),"-hide_banner","-loglevel","error","-n","-f","rawvideo","-pix_fmt","bgr24",
         "-s","1440x500","-r",str(fps/3),"-i","pipe:0","-an","-c:v","libx264","-threads","2",
         "-crf","20","-preset","fast","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    records=[];snaps=[];previous=None
    sample_frames=sorted(set([start,kick,kick+8,kick+15,kick+30,end-1]))
    try:
        for fid in range(start,end):
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f"Decode ended at {fid}")
            proposal=detect_goal_frame(frame);goal=proposal.get("selected")
            status="goal_geometry_unavailable";h=None
            if goal:
                corners=np.array(goal["corners"],float)
                drift=float(np.max(np.linalg.norm(corners-previous,axis=1))) if previous is not None else 0.
                if drift>frame.shape[1]*.045:status="geometry_temporal_jump_rejected"
                else:
                    try:h=fit_goal_plane(corners);status="unverified_goal_plane_candidate";previous=corners
                    except ValueError:status="invalid_quadrilateral"
            canvas=np.full((500,1440,3),(23,26,32),np.uint8)
            left=cv2.resize(frame,(720,405))
            label(canvas,"SOURCE VIDEO + candidate markers",(14,24))
            label(canvas,"GOAL PLANE RECTIFIED - not a reconstructed camera",(734,24))
            evidence=None;gxy=gmap.get(fid);b=bmap.get(fid)
            if gxy is not None:dot(left,[gxy[0]*720/frame.shape[1],gxy[1]*405/frame.shape[0]],(0,220,255))
            if b:dot(left,[b["x"]*720/frame.shape[1],b["y"]*405/frame.shape[0]],(255,240,0))
            if h is not None:
                cv2.polylines(left,[np.rint(corners*[720/frame.shape[1],405/frame.shape[0]]).astype(np.int32)],True,(30,240,80),1)
                right=cv2.warpPerspective(frame,h,(720,405),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(12,12,12))
                cv2.rectangle(right,(60,85),(660,285),(30,240,80),2)
                if b:
                    evidence=ball_projection_evidence(h,b,target_rect=TARGET)
                    dot(right,evidence["rectified_xy"],(255,240,0))
                if gxy is not None:dot(right,project_image_point(h,gxy),(0,220,255))
                # Off-plane pixels are retained, but explicitly not metric or depth-correct.
                label(right,"Off-plane ball/player shapes and locations can be distorted",(8,390),(255,255,255),.47)
            else:
                right=np.full((405,720,3),20,np.uint8)
                label(right,"RECTIFICATION UNAVAILABLE - no carried-forward geometry",(20,180),scale=.48)
            canvas[34:439,:720]=left;canvas[34:439,720:]=right
            label(canvas,f"{cid} | frame {fid} | source {fid/fps:.2f}s | playback 1/3 speed",(14,461),scale=.52)
            label(canvas,"Cyan: ball candidate   Yellow: keeper candidate   No dot = no exact-frame observation",(14,486),scale=.49)
            label(canvas,"OUTCOME: UNKNOWN | depth / full crossing not established",(842,486),(90,180,255),.46)
            proc.stdin.write(canvas.tobytes())
            record={"frame":fid,"geometry_status":status,"goal_corners":goal["corners"] if h is not None else None,
                    "homography":h.tolist() if h is not None else None,"ball_observation":evidence,
                    "keeper_rectified_xy":project_image_point(h,gxy) if h is not None and gxy is not None else None,
                    "outcome":"unknown","source_observations_only":True}
            records.append(record)
            if fid in sample_frames:snaps.append(canvas.copy())
    finally:
        cap.release();proc.stdin.close()
    stderr=proc.stderr.read().decode("utf-8",errors="replace");code=proc.wait()
    if code:raise RuntimeError(stderr)
    probe=cv2.VideoCapture(str(output));count=0;shape=None
    while True:
        ok,decoded=probe.read()
        if not ok:break
        shape=decoded.shape;count+=1
    output_fps=probe.get(cv2.CAP_PROP_FPS);probe.release()
    if count!=len(records) or shape!=(500,1440,3):raise AssertionError("Output failed full decode QA")
    after=sha(video)
    if after!=source_hash:raise AssertionError("Source changed")
    sheet=np.vstack([cv2.resize(v,(1152,400)) for v in snaps])
    cv2.imwrite(str(out/(cid+"_contact_sheet.jpg")),sheet)
    preview=next((v for f,v in zip(sample_frames,snaps) if f==kick+8),snaps[0])
    cv2.imwrite(str(out/(cid+"_preview.jpg")),preview)
    summary={"clip_id":cid,"video_path":str(video),"source_sha256_before":source_hash,"source_sha256_after":after,
        "output_video":str(output),"output_sha256":sha(output),"start_frame":start,"end_frame_exclusive":end,
        "source_fps":fps,"playback_fps":output_fps,"decoded_output_frames":count,
        "geometry_status_counts":dict(Counter(r["geometry_status"] for r in records)),
        "ball_projection_observations":sum(r["ball_observation"] is not None for r in records),
        "projected_ball_center_inside_frames":[r["frame"] for r in records if r["ball_observation"] and r["ball_observation"]["center_projects_inside_mouth"]],
        "source_paths":{"steps":str(step_path),"ball":str(ball_path),"goalkeeper":str(gk_path)},
        "outcome":"unknown","outcome_labels_read":False,"novel_view_reconstruction":False,
        "goal_plane_verified":False,"human_verified":False,"accuracy_validated":False,
        "limits":["Ball and goalkeeper detections are previous unverified candidates, not new validated tracking.",
                  "Homography does not recover ball/player depth, hidden pixels or whole-ball goal-line crossing.",
                  "3:1 display goal rectangle is not metric calibration; corners are automatic unverified proposals.",
                  "Missing geometry is displayed blank; no frame or observation interpolation is fabricated."],
        "frames":records}
    save(out/(cid+".json"),summary)
    return {k:v for k,v in summary.items() if k!="frames"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/goal_plane_rectification_2026-10-10")
    parser.add_argument("--clips",nargs="+",default=["col_eng_p03_cuadrado_col11","col_eng_p06_henderson_eng8"])
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("rectification_tests",ROOT/"tests/test_goal_plane_rectification.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(mod))
    if not result.wasSuccessful():raise AssertionError("Synthetic geometry tests failed")
    rows=[]
    for cid in args.clips:
        rows.append(run_clip(cid,args.output_dir));print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    save(args.output_dir/"summary.json",{"clips":len(rows),"rows":rows,"synthetic_tests_passed":result.testsRun,
        "automatic_outcome_completed":False,"visual_qa":"pending assistant inspection",
        "code_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"src/goal_plane_rectification.py",ROOT/"src/goal_geometry.py"]},
        "sources":["https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html",
                   "https://www.theifab.com/laws/latest/determining-the-outcome-of-a-match/"]})


if __name__=="__main__":main()
