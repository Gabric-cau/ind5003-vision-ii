"""Source-only person ROIs used as ball/limb ambiguity flags, never goal labels."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
import cv2,torch
from ultralytics import YOLO
from src.ball_roi_probe import ROIPredictor
cv2.setNumThreads(2);torch.set_num_threads(3)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-dir',type=Path,
        default=ROOT/'outputs/goal_event_evidence_2026-10-10/detections');args=parser.parse_args()
    base=args.cache_dir
    model=YOLO(str(ROOT/'models/weights/yolo11n_object_detection.pt'))
    for file in sorted(base.glob('*.json')):
        cache=json.loads(file.read_text(encoding='utf-8'))
        if cache.get('status')!='cached_candidates':continue
        dest=base/(cache['clip_id']+'_people.json')
        if dest.exists():raise FileExistsError(dest)
        predictor=ROIPredictor(model,cache['goal_roi']);cap=cv2.VideoCapture(cache['video_path'])
        cap.set(1,cache['start_frame']);frames=[]
        for fid in range(cache['start_frame'],cache['end_frame_exclusive']):
            ok,frame=cap.read()
            if not ok:raise ValueError('Source decode failed')
            frames.append(frame)
        cap.release();rows={}
        for i in range(0,len(frames)):
            results=predictor.predict(frames[i:i+1],imgsz=1280,conf=.12,classes=[0],device='cpu',verbose=False)
            for j,r in enumerate(results):
                rows[str(cache['start_frame']+i+j)]=[{'bbox':list(map(float,b)), 'confidence':float(c)}
                    for b,c in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())]
        dest.write_text(json.dumps({'status':'person_candidates','frames':rows,'source_sha256':cache['source_sha256'],
            'human_verified':False,'outcome_labels_read':False,'batch_size':1},indent=2),encoding='utf-8')
        print(cache['clip_id'],len(rows),flush=True)


if __name__=='__main__':main()
