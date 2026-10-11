"""Render measured ball/HSV keeper candidates beside goal-plane rectification.

No generated camera view, no fabricated observations, no automatic goal label.
Full output decode and original-source hash checks are required.
"""
from pathlib import Path
from collections import Counter
import argparse,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from src.goal_plane_rectification import fit_goal_plane,project_image_point,ball_projection_evidence
from scripts.probe_goal_plane_rectification import label,dot,FFMPEG
cv2.setNumThreads(2)
TARGET=(215.,105.,615.,105.+400./3.)


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(file,out):
    r=load(file);cache=load(Path(r['cache_path']));source=Path(r['video_path']);before=sha(source)
    selected=r['variants']['detector_motion_turn'];track=selected['trajectory'];diag=selected['tracking']
    bmap={d['frame']:d for d in track};kit=r.get('keeper_kit_evidence') or {};gmap=kit.get('frames',{})
    first_post=diag.get('outgoing_start_frame',10**9);start=cache['start_frame'];end=cache['end_frame_exclusive']
    dest=out/(r['clip_id']+'_goal_event_slow.mp4')
    if dest.exists():raise FileExistsError(dest)
    cmd=[str(FFMPEG),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24',
         '-s','1440x550','-r',str(cache['fps']/3),'-i','pipe:0','-an','-c:v','libx264','-threads','2',
         '-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    cap=cv2.VideoCapture(str(source));cap.set(1,start);records=[];previous=None;tiles=[]
    sample=sorted(set([cache['kick_frame']+4,cache['kick_frame']+10,
        first_post-2,first_post,first_post+4,first_post+8]))
    try:
        for f in range(start,end):
            ok,frame=cap.read()
            if not ok:raise ValueError(f'Input decode failed at {f}')
            goal=cache['goal_proposals'][str(f)].get('selected');h=None;state='unavailable'
            if goal:
                corners=np.array(goal['corners'],float)
                jump=float(np.max(np.linalg.norm(corners-previous,axis=1))) if previous is not None else 0
                if jump<frame.shape[1]*.045:
                    try:h=fit_goal_plane(corners,target_rect=TARGET);previous=corners;state='unverified_candidate'
                    except ValueError:state='invalid_geometry'
                else:state='temporal_jump_rejected'
            canvas=np.full((550,1440,3),(23,26,32),np.uint8)
            left=cv2.resize(frame,(720,405))
            right=cv2.warpPerspective(frame,h,(720,405)) if h is not None else np.zeros_like(left)
            label(canvas,'SOURCE VIDEO | measured candidates',(12,25))
            label(canvas,'GOAL PLANE RECTIFIED | not a real front camera',(732,25))
            if h is not None:
                cv2.rectangle(right,tuple(map(round,TARGET[:2])),tuple(map(round,TARGET[2:])),(30,240,80),2)
                cv2.polylines(left,[np.rint(corners*[720/cache['width'],405/cache['height']]).astype(np.int32)],True,(30,240,80),1)
            g=gmap.get(str(f));gxy=None
            if g:
                x0,y0,x1,y1=g['bbox'];gxy=g['center']
                cv2.rectangle(left,(round(x0*720/cache['width']),round(y0*405/cache['height'])),
                    (round(x1*720/cache['width']),round(y1*405/cache['height'])),(0,180,255),1)
                dot(left,[gxy[0]*720/cache['width'],gxy[1]*405/cache['height']],(0,180,255))
                if h is not None:dot(right,project_image_point(h,gxy),(0,180,255))
            tf=cache['camera']['transforms'].get(str(f));ball=bmap.get(f)
            # Small dots are previous measured observations, transported only to
            # compensate the camera. There are no filled-in missing ball frames.
            if tf is not None:
                inv=np.linalg.inv(np.array(tf))
                for d in track:
                    if d['frame']>=f or f-d['frame']>22:continue
                    xy=project_image_point(inv,d['stabilized_xy']);color=(170,80,170) if d['frame']>=first_post else (140,140,40)
                    p=(round(xy[0]*720/cache['width']),round(xy[1]*405/cache['height']))
                    cv2.circle(left,p,2,color,-1)
                    if h is not None:
                        q=project_image_point(h,xy)
                        if q and 0<=q[0]<720 and 0<=q[1]<405:cv2.circle(right,tuple(np.rint(q).astype(int)),2,color,-1)
            evidence=None
            if ball:
                color=(255,80,255) if f>=first_post else (255,240,0)
                dot(left,[ball['x']*720/cache['width'],ball['y']*405/cache['height']],color)
                if h is not None:
                    evidence=ball_projection_evidence(h,ball,target_rect=TARGET);dot(right,evidence['rectified_xy'],color)
                    q=evidence['rectified_xy']
                    if q is None or not (0<=q[0]<720 and 0<=q[1]<405):
                        label(right,'Ball projection outside this viewport',(12,25),(90,180,255),.48)
            label(right,'Off-plane positions/shapes are distorted; depth is unknown',(8,391),scale=.46)
            canvas[35:440,:720]=left;canvas[35:440,720:]=right
            label(canvas,f"{r['clip_id']} | frame {f} | source {f/cache['fps']:.2f}s | playback 1/3",(12,462),scale=.53)
            label(canvas,'Cyan: incoming ball | Magenta: outgoing ball | Orange: keeper kit candidate',(12,486),scale=.51)
            status='OUTGOING CANDIDATE' if ball and f>=first_post else 'INCOMING CANDIDATE' if ball else 'NO CURRENT BALL OBSERVATION'
            label(canvas,status,(12,512),(150,210,250),.5)
            label(canvas,'Small dots = earlier measurements, not interpolated detections',(12,537),scale=.45)
            label(canvas,'OUTCOME: UNKNOWN | whole-ball crossing not established',(745,486),(90,180,255),.48)
            label(canvas,'HSV kit overlap is not proof of physical ball contact',(745,512),scale=.46)
            label(canvas,'Experimental candidates | no human ground truth / accuracy validation',(745,537),scale=.44)
            proc.stdin.write(canvas.tobytes())
            records.append({'frame':f,'geometry_status':state,'ball':evidence,'keeper_kit':g,'outcome':'unknown'})
            if f in sample:tiles.append(canvas.copy())
    finally:cap.release();proc.stdin.close()
    stderr=proc.stderr.read().decode(errors='replace');rc=proc.wait()
    if rc:raise RuntimeError(stderr)
    decode=cv2.VideoCapture(str(dest));n=0;shape=None
    while True:
        ok,im=decode.read()
        if not ok:break
        n+=1;shape=im.shape
    fps=decode.get(cv2.CAP_PROP_FPS);decode.release()
    if n!=end-start or shape!=(550,1440,3):raise AssertionError('Full decode QA failed')
    if sha(source)!=before or before!=cache['source_sha256']:raise AssertionError('Source changed')
    if tiles:
        cv2.imwrite(str(out/(r['clip_id']+'_contact_sheet.jpg')),np.vstack([cv2.resize(t,(1152,440)) for t in tiles]))
        cv2.imwrite(str(out/(r['clip_id']+'_preview.jpg')),tiles[len(tiles)//2])
    result={'clip_id':r['clip_id'],'source':str(source),'source_sha256':before,'output':str(dest),
        'output_sha256':sha(dest),'decoded_frames':n,'playback_fps':fps,'ball_observations':len(track),
        'incoming_observations':diag.get('incoming_observations'),'outgoing_observations':diag.get('outgoing_observations'),
        'geometry_status_counts':dict(Counter(v['geometry_status'] for v in records)),
        'source_observations_only':True,'human_verified':False,'accuracy_validated':False,
        'outcome':'unknown','trajectory_source':str(file),'frames':records}
    save(out/(r['clip_id']+'.json'),result)
    return {k:v for k,v in result.items() if k!='frames'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,default=ROOT/'outputs/goal_event_evidence_2026-10-10/tracking_v6')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/goal_event_evidence_2026-10-10/videos_v6');args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
    for file in sorted(args.input_dir.glob('*.json')):
        if file.name=='summary.json':continue
        rows.append(run(file,args.output_dir));print(json.dumps(rows[-1]),flush=True)
    save(args.output_dir/'summary.json',{'rows':rows,'visual_qa':'pending_assistant_inspection','automatic_outcomes_completed':False})


if __name__=='__main__':main()
