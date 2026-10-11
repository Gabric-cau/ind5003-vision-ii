"""All-39 short-event rule suggestions, source video, and separate review panels."""
from pathlib import Path
import sys,argparse,json,hashlib,subprocess,html
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if sys.version_info[:2]==(3,12):sys.path.insert(0,str(ROOT/'.deps_ball_mask'))
import cv2,numpy as np
from PIL import Image,ImageDraw
from src.goal_plane_rectification import fit_goal_plane,project_image_point
from src.ball_keeper_rebound import associate_keeper_rebound
from src.ball_goal_decision import classify_goal_evidence
from src.project_paths import resolve_project_video
from scripts.review_ball_keeper_rebound import keeper_regions
from scripts.build_ball_detection_seed_evidence import evidence_tile
cv2.setNumThreads(2)
CACHE=ROOT/'outputs/ball_detection_all_2026-10-11/detections'
OLD=ROOT/'outputs/ball_narrow_channel_2026-10-11/results'
FFMPEG=ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pixel_evidence(q,raw):
    x,y=round(q['x']),round(q['y'])
    patch=raw[max(0,y-5):min(raw.shape[0],y+6),max(0,x-5):min(raw.shape[1],x+6)]
    if patch.size==0:return {**q,'rebound_current_white_support':False,'current_white_ratio':0.}
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV);b,g,r=cv2.split(patch.astype(float))
    white=(hsv[:,:,1]<=110)&(hsv[:,:,2]>=135)&(g-np.maximum(r,b)<=25)
    ratio=float(white.mean())
    return {**q,'rebound_current_white_support':ratio>=.20,'current_white_ratio':ratio}


def arrival_evidence(path,c,frames,fingerprints,old):
    seed=old['seed'];origin=np.median([q['stabilized_xy'] for q in seed],axis=0)
    # Goal reference gate uses the same source-independent geometry as the
    # earlier focus phase. It does not estimate the physical goal-line depth.
    gate=old['phase'].get('goal_area_gate_progress_px')
    goal=np.array(load(ROOT/'outputs/ball_mask_review_ready_2026-10-11/inputs'/f'{c["clip_id"]}.json')['goal_center'])
    direction=goal-origin;direction=direction/max(1e-8,np.linalg.norm(direction))
    baseline=c['goal_proposals'].get(str(c['kick_frame']),{}).get('selected')
    result=[]
    for p in path:
        f=p['frame'];proposal=c['goal_proposals'].get(str(f),{}).get('selected');raw=frames.get(f)
        if proposal is None or raw is None:continue
        matrix=fit_goal_plane(proposal['corners'],target_rect=(0.,0.,1.,1.))
        center=project_image_point(matrix,[p['x'],p['y']])
        rx=min(p.get('width',10),p.get('height',10))/2
        footprint=[project_image_point(matrix,[p['x']+dx,p['y']+dy]) for dx,dy in
                   [(-rx,0),(rx,0),(0,-rx),(0,rx)]]
        u,v=center;radx=max(abs(z[0]-u) for z in footprint);rady=max(abs(z[1]-v) for z in footprint)
        inside=0<=u<=1 and 0<=v<=1
        # Below the posts may still be foreground ground-plane perspective;
        # only above the bar or beyond either side is an outside-mouth cue.
        outside=u+radx<-.04 or u-radx>1.04 or v+rady<-.04
        supported=True
        if baseline:
            tf=c['camera']['transforms'][str(f)];kt=c['camera']['transforms'][str(c['kick_frame'])]
            reference=np.array([project_image_point(kt,q) for q in baseline['corners']])
            current=np.array([project_image_point(tf,q) for q in proposal['corners']])
            supported=float(np.mean(np.linalg.norm(current-reference,axis=1)))<.2*(baseline['bbox'][2]-baseline['bbox'][0])
        cue=pixel_evidence(p,raw)
        progress=float(np.dot(np.array(p['stabilized_xy'])-origin,direction))
        result.append({'frame':f,'detection_id':p['detection_id'],'normalized_mouth_xy':center,
            'normalized_radius':[radx,rady],'inside_mouth':bool(inside),'clearly_outside_mouth':bool(outside),
            'identity_support':cue['current_white_ratio']>=.15,'current_white_ratio':cue['current_white_ratio'],
            'geometry_supported':bool(supported),'arrival_progress_supported':bool(gate is not None and progress>=gate-20),
            'independent_image_key':fingerprints[str(f)],'projection_only':True,
            'source_xy':[p['x'],p['y']]})
    return result


def process(cid,index,out,freshdir):
    c,old,fresh=load(CACHE/f'{cid}.json'),load(OLD/f'{cid}.json'),load(freshdir/f'{cid}.json')
    c['video_path']=str(resolve_project_video(ROOT,c['video_path'],cid))
    assert sha(c['video_path'])==c['source_sha256']==old['source_sha256']==fresh['source_sha256']
    start,end=fresh['start_frame'],fresh['end_frame_exclusive'];boundary=old['phase']['main_end_frame']
    cap=cv2.VideoCapture(c['video_path']);cap.set(1,start);frames={};fingerprints={};prev=None;last_hash=None
    for f in range(start,end):
        ok,im=cap.read();assert ok
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);duplicate=prev is not None and np.mean(cv2.absdiff(gray,prev))<.35
        last_hash=last_hash if duplicate else hashlib.sha256(im.tobytes()).hexdigest()
        frames[f]=im;fingerprints[str(f)]=last_hash;prev=gray
    cap.release()
    pool={f:[pixel_evidence(p,frames[int(f)]) for p in qs] if int(f) in frames else list(qs)
          for f,qs in c['detections'].items()}
    for f,v in fresh['frames'].items():pool.setdefault(f,[]).extend(pixel_evidence(p,frames[int(f)]) for p in v['balls'])
    prefix=[p for p in old['trajectory'] if p['frame']<=boundary]
    regions,_=keeper_regions(c,fresh)
    if len(prefix)>=3:
        path,rebound=associate_keeper_rebound(prefix,pool,regions,c['fps'],frame_fingerprints=fingerprints)
    else:path,rebound=prefix,{'turn_confirmed':False,'reason':'insufficient_incoming_prefix'}
    if not rebound['turn_confirmed']:path=[p for p in old['trajectory'] if p['frame']<end]
    observations=arrival_evidence(path,c,frames,fingerprints,old)
    decision=classify_goal_evidence(observations,boundary,rebound)
    result={'clip_id':cid,'event_id':index,'source_video_path':c['video_path'],
        'source_sha256':sha(c['video_path']),'source_cache_sha256':sha(CACHE/f'{cid}.json'),
        'fresh_candidates_sha256':sha(freshdir/f'{cid}.json'),'old_track_sha256':sha(OLD/f'{cid}.json'),
        'trajectory':path,'rebound':rebound,'rule_decision':decision,'arrival_evidence':observations,
        'short_window':[start,end],'boundary_frame':boundary,'fps':c['fps'],
        'visual_review':{'status':'pending','outcome':'unknown','human_verified':False},
        'manual_ball_points_used':False,'outcome_labels_used_for_algorithm':False,'accuracy_validated':False}
    assert path[:len(prefix)]==prefix
    save(out/'results'/f'{cid}.json',result)
    # A silent source replay: only the event window, with old and new candidates
    # separate. No event label is rendered before the visual review is written.
    dest=out/'media'/f'{cid}.mp4'
    proc=subprocess.Popen([str(FFMPEG),'-hide_banner','-loglevel','error','-n','-f','rawvideo',
      '-pix_fmt','bgr24','-s','1120x640','-r',str(c['fps']/3),'-i','pipe:0','-an',
      '-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],
      stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    picked={p['frame']:p for p in path};oldpts={p['frame']:p for p in old['trajectory']};tiles=[]
    for f,raw in frames.items():
        canvas=np.full((640,1120,3),25,np.uint8);scene=cv2.resize(raw,(900,506))
        scale=np.array([900/c['width'],506/c['height']]);goal=c['goal_proposals'].get(str(f),{}).get('selected')
        if goal:cv2.polylines(scene,[np.rint(np.array(goal['corners'])*scale).astype(np.int32)],True,(210,185,50),1,cv2.LINE_AA)
        for q,color,radius,thickness in [(oldpts.get(f),(255,150,50),10,1),(picked.get(f),(65,255,75),7,2)]:
            if q:cv2.circle(scene,tuple(np.rint(np.array([q['x'],q['y']])*scale).astype(int)),radius,color,thickness,cv2.LINE_AA)
        canvas[45:551,:900]=scene
        cv2.putText(canvas,f'Event {index:02d} | {cid} | source frame {f}',(12,27),0,.55,(240,240,240),1,cv2.LINE_AA)
        q=picked.get(f)
        if q:
            tile=evidence_tile(Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)),q,f'Event {index:02d} | f{f}')
            canvas[75:273,900:1120]=np.asarray(tile.resize((220,198)))[:,:,::-1]
            if boundary-1<=f<=boundary+10 and not (tiles and fingerprints[str(f)]==tiles[-1][2]):tiles.append((f,tile,fingerprints[str(f)]))
        else:cv2.putText(canvas,'No selected observation',(910,120),0,.42,(220,220,220),1)
        cv2.putText(canvas,'Green: associated candidate | Blue: previous candidate | Cyan: goal-frame proposal',(12,586),0,.53,(230,230,230),1)
        cv2.putText(canvas,'Short event window | Predictions and 2D projections do not establish whole-ball crossing',(12,613),0,.5,(230,230,230),1)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    for j in range(0,len(tiles),8):
        group=tiles[j:j+8];panel=Image.new('RGB',(1200,270*((len(group)+3)//4)),(25,25,25))
        for k,(_,tile,_) in enumerate(group):panel.paste(tile,(k%4*300,k//4*270))
        panel.save(out/'media'/f'event_{index:02d}_selected_{j//8+1}.jpg',quality=94)
    # Preserve the exact raw panels being independently visually reviewed.
    rawpanel=ROOT/'outputs/ball_goal_event_audit_2026-10-11/media'/f'event_{index:02d}.jpg'
    (out/'media'/rawpanel.name).write_bytes(rawpanel.read_bytes())
    vc=cv2.VideoCapture(str(dest));decoded=0
    while True:
        ok,_=vc.read()
        if not ok:break
        decoded+=1
    vc.release();assert decoded==end-start
    print(f'{index:02d}/39 {cid}: {decision["rule_outcome"]}; {rebound["reason"]}',flush=True)
    return {'event_id':index,'clip_id':cid,'rule_outcome':decision['rule_outcome'],
            'reason':decision['reason'],'rebound_supported':rebound['turn_confirmed'],
            'video_decoded_frames':decoded,'visual_outcome':'unknown'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fresh-dir',type=Path,required=True)
    parser.add_argument('--new-inference-clips',type=int,choices=range(40),default=0)
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(exist_ok=False)
    for folder in ['results','media','source_snapshot']:(out/folder).mkdir()
    ids=load(ROOT/'outputs/ball_mask_review_ready_2026-10-11/protocol.json')['canonical_clips']
    assert len(ids)==len(set(ids))==39
    files=[Path(__file__),ROOT/'src/ball_goal_decision.py',ROOT/'src/ball_keeper_rebound.py',ROOT/'src/project_paths.py',
       ROOT/'scripts/review_ball_keeper_rebound.py',ROOT/'scripts/cache_ball_keeper_contact.py']
    for p in files:(out/'source_snapshot'/p.name).write_bytes(p.read_bytes())
    save(out/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),'canonical_clips':ids,
        'fresh_candidates_dir':str(args.fresh_dir.resolve()),'outcome_labels_used_for_algorithm':False,
        'manual_ball_points_used':False,'visual_review_used_for_algorithm':False,
        'rule_labels_are_suggestions':True,'human_verified':False,'accuracy_validated':False,
        'new_neural_inference_clips':args.new_inference_clips,'reused_fresh_neural_inference_clips':39-args.new_inference_clips,
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    rows=[process(cid,i,out,args.fresh_dir.resolve()) for i,cid in enumerate(ids,1)]
    save(out/'summary.json',{'rows':rows,'clips':39,'all_sources_decoded':True,'accuracy_validated':False,
        'human_verified':False,'visual_review_status':'pending'})


if __name__=='__main__':main()
