"""Reproducible short keeper-rebound experiment using automatic candidates."""
from pathlib import Path
import sys, argparse, json, hashlib, html, subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if sys.version_info[:2]==(3,12):sys.path.insert(0,str(ROOT/'.deps_ball_mask'))
import numpy as np
import cv2
from PIL import Image, ImageDraw
from src.goal_plane_rectification import project_image_point
from src.ball_keeper_rebound import associate_keeper_rebound
from src.ball_direction_corridor import corridor_geometry, established_radius
from scripts.build_ball_detection_seed_evidence import evidence_tile


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def keeper_regions(cache,fresh):
    """Choose the pre-shot person inside the goal, then follow that person.

    Detector boxes and short held boxes remain unverified proximity evidence.
    No outcome, manually drawn mask, or selected rebound point is used.
    """
    kick=cache['kick_frame'];previous=None;last_good=None;regions={};boxes={}
    for f in range(fresh['start_frame'],fresh['end_frame_exclusive']):
        goal=cache['goal_proposals'].get(str(f),{}).get('selected')
        tf=cache['camera']['transforms'].get(str(f))
        if not goal or tf is None:continue
        gx,gy,gb,gd=goal['bbox'];gw,gh=gb-gx,gd-gy
        target=project_image_point(tf,[(gx+gb)/2,gd-.25*gh])
        choices=[]
        for p in fresh['frames'][str(f)]['people']:
            x,y,b,d=p['bbox'];h=d-y;w=b-x
            if not (gx-.1*gw<=(x+b)/2<=gb+.1*gw and gy+.4*gh<d<gd+.12*gh):continue
            if max(h,w)<.24*gh:continue
            poly=[project_image_point(tf,q) for q in ([x,y],[b,y],[b,d],[x,d])]
            center=np.mean(poly,axis=0)
            if previous is None:
                dist=float(np.linalg.norm(center-target))
                if f>kick or dist>.4*gh:continue
            else:
                dist=float(np.linalg.norm(center-np.mean(previous,axis=0)))
                if dist>65:continue
            choices.append((dist-5*p['confidence'],poly,p))
        held=False
        if choices:
            _,previous,p=min(choices,key=lambda q:q[0]);last_good=f
            boxes[str(f)]=p
        elif previous is not None and f-last_good<=4:
            held=True
        else:continue
        regions[str(f)]=[{'kind':'keeper_candidate','polygon':previous,
            'source':'automatic_person_track_short_hold' if held else 'fresh_person_detection',
            'geometry_last_measured_frame':last_good,'held_prediction':held,
            'physical_contact_established':False}]
    return regions,boxes


def run(cid,freshdir,out):
    oldfile=ROOT/'outputs/ball_narrow_channel_2026-10-11/results'/f'{cid}.json'
    cachefile=ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json'
    freshfile=freshdir/f'{cid}.json'
    old,c,fresh=load(oldfile),load(cachefile),load(freshfile)
    assert sha(c['video_path'])==fresh['source_sha256']==old['source_sha256']
    boundary=old['phase']['main_end_frame']
    incoming=[q for q in old['trajectory'] if q['frame']<=boundary]
    regions,boxes=keeper_regions(c,fresh)
    pool={f:list(qs) for f,qs in c['detections'].items()}
    for f,v in fresh['frames'].items():pool.setdefault(f,[]).extend(v['balls'])
    cap=cv2.VideoCapture(c['video_path']);start=max(c['start_frame'],c['kick_frame']-5)
    end=fresh['end_frame_exclusive'];cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    frames={};fingerprints={};previous_gray=None;last_fingerprint=None
    for f in range(start,end):
        ok,im=cap.read();assert ok
        frames[f]=im;gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        duplicate=previous_gray is not None and float(np.mean(cv2.absdiff(gray,previous_gray)))<.35
        fingerprint=last_fingerprint if duplicate else hashlib.sha256(im.tobytes()).hexdigest()
        fingerprints[str(f)]=fingerprint;previous_gray=gray;last_fingerprint=fingerprint
    cap.release()
    # A conservative current-pixel cue for this white-ball dataset. This only
    # filters new rebound hypotheses; it never removes incoming observations.
    # Yellow gloves/green kit can receive a high sports-ball detector score.
    for f,qs in pool.items():
        raw=frames.get(int(f))
        if raw is None:continue
        for q in qs:
            x,y=round(q['x']),round(q['y'])
            patch=raw[max(0,y-5):min(raw.shape[0],y+6),max(0,x-5):min(raw.shape[1],x+6)]
            if patch.size==0:continue
            hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV);b,g,r=cv2.split(patch.astype(float))
            mask=(hsv[:,:,1]<=110)&(hsv[:,:,2]>=135)&(g-np.maximum(r,b)<=25)
            ratio=float(mask.mean())
            q['rebound_current_white_ratio']=ratio
            q['rebound_current_white_support']=ratio>=.20
    path,diag=associate_keeper_rebound(incoming,pool,regions,c['fps'],frame_fingerprints=fingerprints)
    if not diag['turn_confirmed']:
        stop=old['phase']['focus_end_frame']
        path=[q for q in old['trajectory'] if q['frame']<=stop]
    else:stop=min(end-1,diag['context_end_frame'])
    result={'clip_id':cid,'trajectory':path,'rebound':diag,'boundary_frame':boundary,
        'source_video_path':c['video_path'],'source_sha256':sha(c['video_path']),
        'old_result_sha256':sha(oldfile),'fresh_candidates_sha256':sha(freshfile),
        'keeper_regions':regions,'new_model_inference':True,'human_verified':False,
        'accuracy_validated':False,'outcome_labels_used':False,'frame_fingerprints':fingerprints,
        'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__),ROOT/'src/ball_keeper_rebound.py',ROOT/'scripts/cache_ball_keeper_contact.py']}}
    # Preserve the prefix bit for bit; predicted contact is never appended.
    assert path[:len(incoming)]==incoming
    save(out/'results'/f'{cid}.json',result)
    picked={q['frame']:q for q in path};previous={q['frame']:q for q in old['trajectory']}
    ffmpeg=ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    video=out/'media'/f'{cid}.mp4'
    proc=subprocess.Popen([str(ffmpeg),'-hide_banner','-loglevel','error','-n','-f','rawvideo',
        '-pix_fmt','bgr24','-s','1120x640','-r',str(c['fps']/3),'-i','pipe:0','-an',
        '-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    tiles=[]
    for f in range(start,stop+1):
        raw=frames[f];scene=cv2.resize(raw,(900,506));scale=np.array([900/c['width'],506/c['height']])
        canvas=np.full((640,1120,3),25,np.uint8)
        if f in previous:
            q=previous[f];cv2.circle(scene,tuple(np.rint([q['x']*scale[0],q['y']*scale[1]]).astype(int)),11,(255,160,50),1)
        q=picked.get(f)
        if q:
            cv2.circle(scene,tuple(np.rint([q['x']*scale[0],q['y']*scale[1]]).astype(int)),7,(60,255,70),2)
        # Draw a corridor only from prior actual observations. Repeated source
        # images are not counted as a new confirmation or motion update.
        prior=[a for a in path if a['frame']<f]
        if diag['turn_confirmed'] and f>boundary:
            # Freeze the incoming lane in the contact interval. The outgoing
            # lane uses only prior outgoing observations after confirmation.
            prior=([a for a in prior if a['frame']>boundary]
                   if f>diag['confirmation_frame'] else [])
        if len(prior)>=2 and f-prior[-1]['frame']<=4:
            a,b=prior[-2:];v=(np.array(b['stabilized_xy'])-a['stabilized_xy'])/(b['frame']-a['frame'])
            predicted=np.array(b['stabilized_xy'])+v*(f-b['frame'])
            inv=np.linalg.inv(c['camera']['transforms'][str(f)])
            rawp=project_image_point(inv,predicted)
            if rawp is not None:
                cv2.circle(scene,tuple(np.rint(np.array(rawp)*scale).astype(int)),12,(0,170,255),1)
            if np.linalg.norm(v)>1e-8:
                g=corridor_geometry(b['stabilized_xy'],v,f-b['frame'],established_radius(prior),max(20.,np.linalg.norm(v)*1.5))
                poly=[project_image_point(inv,p) for p in g['polygon_stabilized_xy']]
                if all(p is not None for p in poly):
                    cv2.polylines(scene,[np.rint(np.array(poly)*scale).astype(np.int32)],True,(210,190,35),1,cv2.LINE_AA)
        canvas[45:551,:900]=scene
        cv2.putText(canvas,f'{cid} | source frame {f} | kick {c["kick_frame"]}',(12,27),0,.62,(245,245,245),1,cv2.LINE_AA)
        if q:
            tile=evidence_tile(Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)),q,f'{cid} | f{f}')
            crop=np.asarray(tile.resize((220,198)))[:,:,::-1];canvas[80:278,900:1120]=crop
        else:
            cv2.putText(canvas,'No observation',(910,130),0,.5,(220,220,220),1,cv2.LINE_AA)
        status='Incoming observation' if f<=boundary else 'Short contact / rebound evidence'
        cv2.putText(canvas,status,(12,579),0,.65,(60,255,70),1,cv2.LINE_AA)
        cv2.putText(canvas,'Green: observation | Blue: previous | Orange: forecast | Cyan: search lane',(12,610),0,.53,(235,235,235),1,cv2.LINE_AA)
        proc.stdin.write(canvas.tobytes())
        if q and f>boundary:tiles.append((f,tile))
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    for j in range(0,len(tiles),8):
        group=tiles[j:j+8];panel=Image.new('RGB',(1200,270*((len(group)+3)//4)),(25,25,25))
        for k,(f,tile) in enumerate(group):panel.paste(tile,(k%4*300,k//4*270))
        panel.save(out/'media'/f'{cid}_evidence_{j//8+1}.jpg',quality=95)
    check=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,_=check.read()
        if not ok:break
        decoded+=1
    check.release();assert decoded==stop-start+1
    print(cid,diag['reason'],'support',diag.get('support_frames'),'end',stop,'points',[(q['frame'],q['detection_id']) for q in path if q['frame']>boundary],flush=True)
    return {'clip_id':cid,'turn_confirmed':diag['turn_confirmed'],'decoded_frames':decoded,
            'stop_frame':stop,'support_frames':diag.get('support_frames'),
            'last_observed_frame':diag.get('last_observed_frame'),'incoming_prefix_preserved':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--clips',nargs='+',required=True)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(exist_ok=False)
    (out/'media').mkdir();(out/'results').mkdir()
    checks=[run(cid,args.fresh_dir.resolve(),out) for cid in args.clips]
    save(out/'validation.json',{'clips':checks,'accuracy_validated':False,'human_verified':False,
        'scope':'Targeted source replay and decode QA, not all-39 accuracy evaluation'})
    sections=[]
    for ck in checks:
        cid=ck['clip_id'];r=load(out/'results'/f'{cid}.json');d=r['rebound']
        text=('接触后重新关联到连续换向候选；绿色是实际候选观测。' if d['turn_confirmed'] else '未确认新的守门员反弹分支，保留上版轨迹。')
        pics=''.join(f'<img src="media/{p.name}" alt="原像素局部核对">' for p in (out/'media').glob(f'{cid}_evidence_*.jpg'))
        sections.append(f'<section><h2>{html.escape(cid)}</h2><p>{text} 支持帧：{d.get("support_frames",[])}。观测结束：{ck["last_observed_frame"]}，短窗口结束：{ck["stop_frame"]}。</p><video controls preload="metadata" src="media/{cid}.mp4"></video>{pics}<p><a href="results/{cid}.json">关联与来源记录</a></p></section>')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>守门员反弹关联核对</title><style>body{background:#151a20;color:#eee;font:18px/1.6 system-ui;margin:25px}section{border:1px solid #58606c;padding:18px;margin:20px 0}video,img{display:block;max-width:100%;margin:15px 0}a{color:#9ed7ff}</style><h1>守门员接触后的短反弹窗口</h1><p>修正目标：来球通道不能继续越过扑救位置，接触后的方向须由独立观测重新确认。蓝圈保留旧版位置便于比较；橙圈仅预测，不能当作已找到球。</p><p>使用自动守门员候选、同一原视频与新的局部检测，不输入人工球坐标或进球标签。这里只核对短反弹轨迹；2D接触预测与球门投影都不自动证明进球或没进。没有完整人工逐帧标注，不报告准确率。</p>'''+''.join(sections)+'</html>'
    (out/'report.html').write_text(document,encoding='utf-8')


if __name__=='__main__':main()
