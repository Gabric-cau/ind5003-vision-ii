"""Fresh goal-area ball/person candidates for short contact/rebound review."""
from pathlib import Path
import argparse
import sys
import json
import hashlib
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import torch
from ultralytics import YOLO
from src.goal_plane_rectification import project_image_point
from src.project_paths import resolve_project_video
from src.runtime import add_runtime_arguments, resolve_device, inference_provenance
from scripts.run_ball_direction_corridor_review import load,save,sha

cv2.setNumThreads(2);torch.set_num_threads(3)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--clips',nargs='+',default=['col_eng_p09_bacca_col7'])
    add_runtime_arguments(parser,ffmpeg=False)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'data/raw_clips')
    parser.add_argument('--ball-cache-dir',type=Path,default=ROOT/'outputs/ball_detection_all_2026-10-11/detections')
    parser.add_argument('--track-dir',type=Path,default=ROOT/'outputs/ball_narrow_channel_2026-10-11/results')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    device=resolve_device(args.device,torch)
    weights=ROOT/'models/weights/yolo11n_object_detection.pt';model=YOLO(str(weights))
    for cid in args.clips:
        dest=out/f'{cid}.json'
        if dest.exists():raise FileExistsError(dest)
        cache_path=args.ball_cache_dir/f'{cid}.json';track_path=args.track_dir/f'{cid}.json'
        c=load(cache_path)
        c['video_path']=str(resolve_project_video(ROOT,c['video_path'],cid,args.source_dir))
        if sha(c['video_path']) != c['source_sha256']:
            raise ValueError('Ball cache belongs to different source video bytes')
        old=load(track_path)
        if old['source_sha256'] != c['source_sha256']:
            raise ValueError('Incoming track belongs to a different source')
        start=max(c['start_frame'],c['kick_frame']-5)
        end=min(c['end_frame_exclusive'],old['phase']['main_end_frame']+round(.65*c['fps'])+1)
        goal=c['goal_proposals'].get(str(c['kick_frame']),{}).get('selected')
        if goal:
            x0,y0,x1,y1=goal['bbox'];w=x1-x0;h=y1-y0
            roi=[max(0,int(x0-.2*w)),max(0,int(y0-.25*h)),min(c['width'],int(x1+.65*w)),min(c['height'],int(y1+.65*h))]
        else:roi=c['goal_roi']
        x0,y0,x1,y1=roi;cap=cv2.VideoCapture(c['video_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        frames={};batch=[];batch_ids=[]
        def infer():
            pred=model.predict(batch,classes=[0,32],conf=.01,imgsz=960,device=device,verbose=False)
            for f,result in zip(batch_ids,pred):
                people=[];balls=[];tf=c['camera']['transforms'].get(str(f))
                for j,box in enumerate(result.boxes):
                    cls=int(box.cls.item());conf=float(box.conf.item())
                    xa,ya,xb,yb=box.xyxy[0].tolist();xa+=x0;xb+=x0;ya+=y0;yb+=y0
                    if cls==0 and conf>=.18:
                        people.append({'bbox':[xa,ya,xb,yb],'confidence':conf,'class_id':0})
                    elif cls==32 and tf is not None:
                        x,y=(xa+xb)/2,(ya+yb)/2
                        balls.append({'frame':f,'x':x,'y':y,'width':xb-xa,'height':yb-ya,'confidence':conf,
                          'sources':['fresh_contact_roi'],'detection_id':f'{f}:contact:{j}',
                          'stabilized_xy':project_image_point(tf,[x,y])})
                frames[str(f)]={'people':people,'balls':balls}
            batch.clear();batch_ids.clear()
        for f in range(start,end):
            ok,raw=cap.read();assert ok
            batch.append(raw[y0:y1,x0:x1]);batch_ids.append(f)
            if len(batch)==4:infer()
        if batch:infer()
        cap.release()
        save(dest,{'clip_id':cid,'source_sha256':sha(c['video_path']),'start_frame':start,'end_frame_exclusive':end,
          'roi':roi,'frames':frames,'weights_sha256':sha(weights),'image_size':960,'classes':[0,32],
          'created_at':datetime.now(timezone.utc).isoformat(),'new_model_inference':True,'outcome_labels_used':False,
          'inference_scope':'short_goal_area_contact_window','complete_raw_to_all_reconstruction':False,
          'runtime':inference_provenance(args.device,device,[weights],[c['video_path']],torch),
          'ball_cache_sha256':sha(cache_path),'track_sha256':sha(track_path),
          'manual_ball_points_used':False,'human_verified':False,'code_sha256':sha(__file__)})
        print(f'{cid}: fresh contact ROI {start}..{end-1}',flush=True)


if __name__=='__main__':main()
