"""Matched-input ablation: retain upright tall-digit crops before PP-OCRv4 rec."""
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'.deps_jersey'))
import cv2
from rapidocr_onnxruntime import RapidOCR
from scripts.run_jersey_ocr_worker import read,save,sha
from src.upright_text_crop import warp_text_upright


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/jersey_retrieval_2026-10-11_v1')
    p.add_argument('--resume',action='store_true');p.add_argument('--clips',nargs='*');a=p.parse_args();out=a.output_dir
    cfg={'worker_sha256':sha(__file__),'crop_code_sha256':sha(ROOT/'src/upright_text_crop.py'),
          'models':{f.name:sha(f) for f in sorted((ROOT/'.deps_jersey/rapidocr_onnxruntime/models').glob('*.onnx'))},
          'engine':'PP-OCRv4 detector + recognizer; no automatic 90deg crop rotation',
          'det_limit_side_len':480,'det_limit_type':'max','text_score':.5,'baseline_protocol_sha256':sha(out/'ocr_protocol.json'),
          'expected_values_provided_to_OCR':False,'ablation_note':'Same 1912 selected crop files; direct detector and perspective warp retain tall-box orientation.'}
    path=out/'upright_ocr_protocol.json'
    if path.exists():
        if read(path)!=cfg:raise ValueError('Protocol changed')
    else:save(path,cfg)
    engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1,det_limit_side_len=480,det_limit_type='max',use_cls=False,text_score=.5)
    cv2.setNumThreads(2);total=0;start=time.monotonic()
    for path in sorted((out/'gallery').glob('*.json')):
        cid=path.stem
        if a.clips and cid not in a.clips:continue
        target=out/'ocr_upright'/f'{cid}.json'
        if target.exists():
            if not a.resume:raise FileExistsError(target)
            existing=read(target)
            if existing['gallery_sha256']!=sha(path) or existing['protocol_sha256']!=sha(out/'upright_ocr_protocol.json'):raise ValueError('Cache mismatch')
            continue
        gallery=read(path);results=[]
        for s in gallery['shots']:
            for t in s['tracklets']:
                rows=[]
                for k in t['keyframes']:
                    image=out/k['image']
                    if sha(image)!=k['image_sha256']:raise ValueError('Crop changed')
                    data=cv2.imread(str(image));h,w=data.shape[:2]
                    scale=min(2.,640/max(h,w));input_img=cv2.resize(data,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                    boxes,_=engine.text_det(input_img);tokens=[]
                    boxes=[] if boxes is None else list(boxes)
                    rects=[warp_text_upright(input_img,b) for b in boxes]
                    valid=[(b,c) for b,c in zip(boxes,rects) if c is not None]
                    recognized=engine.text_rec([c for _,c in valid])[0] if valid else []
                    for (box,crop),(text,confidence) in zip(valid,recognized):
                        if confidence<.5:continue
                        points=[[float(x)/scale,float(y)/scale] for x,y in box]
                        th=max(v[1] for v in points)-min(v[1] for v in points)
                        tokens.append({'text':text,'confidence':float(confidence),'box_in_crop':points,
                                       'within_torso':bool(th>=.035*h),'tall_box_90_degree_rotation_avoided':bool(crop.shape[0]/crop.shape[1]>=1.5)})
                    rows.append({'frame':k['frame'],'time_sec':k['time_sec'],'image':k['image'],'tokens':tokens});total+=1
                results.append({'segment_id':s['segment_id'],'tracklet_id':t['tracklet_id'],'observations':rows})
        save(target,{'clip_id':cid,'gallery_sha256':sha(path),'protocol_sha256':sha(out/'upright_ocr_protocol.json'),
                    'tracklets':results,'identity_verified':False,'human_verified':False})
        print(json.dumps({'clip':cid,'crops_processed_this_run':total,'elapsed_sec':round(time.monotonic()-start,1)}),flush=True)


if __name__=='__main__':main()
