"""Render ALL supported or conflicting text tracklets for explicit AI visual QA."""
from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_jersey_gallery import DEFAULT
from scripts.run_camera_partial_pose_experiment import read,save


def fit(image,w,h):
    scale=min(w/image.shape[1],h/image.shape[0]);small=cv2.resize(image,None,fx=scale,fy=scale)
    canvas=np.full((h,w,3),240,np.uint8);y=(h-small.shape[0])//2;x=(w-small.shape[1])//2
    canvas[y:y+small.shape[0],x:x+small.shape[1]]=small;return canvas


def main():
    out=DEFAULT/'visual_qa';out.mkdir(exist_ok=True)
    rows=[r for r in read(DEFAULT/'evaluation_v1/retrieval_index.json')['tracklets'] if r['upright_query']['status']!='unknown']
    records=[]
    for offset in range(0,len(rows),5):
        canvas=np.full((5*360,1200,3),245,np.uint8)
        for j,r in enumerate(rows[offset:offset+5]):
            e=r['upright_evidence'];number=e['number'];frames=[]
            if number is not None:
                frames=[s['frame'] for v in e['ranked_number_votes'] if v['number']==number for s in v['samples']]
            if not frames:frames=r['upright_query']['name_hit_frames']
            if not frames:frames=[k['frame'] for k in r['keyframes']]
            frames=sorted(frames);picked=[frames[0],frames[-1]]
            fid=picked[0];cap=cv2.VideoCapture(str(ROOT/'data/raw_clips'/f'{r["clip_id"]}.mp4'));cap.set(1,fid);ok,im=cap.read();cap.release()
            if not ok:raise RuntimeError('decode')
            k=next(k for k in r['keyframes'] if k['frame']==fid);b=list(map(int,k['person_bbox']))
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,220,255),3)
            label=f'#{offset+j:02} {r["clip_id"]} s{r["segment_id"]} t{r["tracklet_id"]} expected={r["expected_query"]["number"]} OCR={number}'
            cv2.putText(canvas,label,(10,j*360+23),cv2.FONT_HERSHEY_SIMPLEX,.54,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(canvas,r['upright_query']['status'],(10,j*360+48),cv2.FONT_HERSHEY_SIMPLEX,.54,(0,0,0),1,cv2.LINE_AA)
            canvas[j*360+65:j*360+345,:600]=fit(im,600,280)
            names=[]
            for ci,fid in enumerate(picked):
                k=next(k for k in r['keyframes'] if k['frame']==fid);image=cv2.imread(str(DEFAULT/k['image']));thumb=fit(image,300,280)
                cv2.putText(thumb,f'frame {fid}',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
                canvas[j*360+65:j*360+345,600+ci*300:900+ci*300]=thumb;names.append(k['image'])
            records.append({'qa_index':offset+j,'clip_id':r['clip_id'],'segment_id':r['segment_id'],'tracklet_id':r['tracklet_id'],
                  'expected_number_metadata':r['expected_query']['number'],'ocr_number':number,'evidence_images':names,'frames':picked})
        path=out/f'evidence_{offset:02}_{min(offset+4,len(rows)-1):02}.jpg'
        cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(path))
    save(out/'index.json',{'records':records,'human_verified':False,'scope':'all non-unknown text tracklets; two OCR-supporting crops and one source context frame; not continuous review'})
    print({'qa_records':len(records)})


if __name__=='__main__':main()
