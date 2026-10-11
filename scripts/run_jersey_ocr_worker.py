"""Python 3.12 OCR worker using local PP-OCRv4; never reads expected identities."""
from pathlib import Path
import argparse,json,sys,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.append(str(ROOT/'.deps_jersey'))
import cv2
from rapidocr_onnxruntime import RapidOCR


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/jersey_retrieval_2026-10-11_v1')
    p.add_argument('--resume',action='store_true');p.add_argument('--clips',nargs='*')
    a=p.parse_args();out=a.output_dir
    cfg={'worker_sha256':sha(__file__),'engine':'rapidocr_onnxruntime 1.4.4 / generic PP-OCRv4; not SoccerNet-trained',
        'models':{f.name:sha(f) for f in sorted((ROOT/'.deps_jersey/rapidocr_onnxruntime/models').glob('*.onnx'))},
        'det_limit_side_len':480,'det_limit_type':'max','use_cls':False,'text_score':.5,
        'input':'target-blind torso crops','rotation':'no automatic 180 rotation; preserve jersey orientation',
        'human_labels_read':False,'expected_number_or_name_provided_to_OCR':False}
    protocol=out/'ocr_protocol.json'
    if protocol.exists():
        if read(protocol)!=cfg:raise ValueError('Protocol changed')
    else:save(protocol,cfg)
    engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1,det_limit_side_len=480,
                    det_limit_type='max',use_cls=False,text_score=.5)
    cv2.setNumThreads(2);total=0;start=time.monotonic()
    for path in sorted((out/'gallery').glob('*.json')):
        cid=path.stem
        if a.clips and cid not in a.clips:continue
        target=out/'ocr'/f'{cid}.json'
        if target.exists():
            if not a.resume:raise FileExistsError(target)
            cached=read(target)
            if cached['gallery_sha256']!=sha(path) or cached['protocol_sha256']!=sha(protocol):raise ValueError('OCR cache changed')
            continue
        gallery=read(path);results=[]
        for s in gallery['shots']:
            for t in s['tracklets']:
                rows=[]
                for k in t['keyframes']:
                    image=out/k['image']
                    if sha(image)!=k['image_sha256']:raise ValueError('Crop changed')
                    data=cv2.imread(str(image));h,w=data.shape[:2]
                    # Upscaling is interpolation, not detail generation. Cap image size.
                    scale=min(2.,640/max(h,w));input_img=cv2.resize(data,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                    result,_=engine(input_img,use_cls=False);tokens=[]
                    for box,text,confidence in result or []:
                        points=[[float(x)/scale,float(y)/scale] for x,y in box]
                        th=max(v[1] for v in points)-min(v[1] for v in points)
                        tokens.append({'text':text,'confidence':float(confidence),'box_in_crop':points,
                                       'within_torso':bool(th>=.035*h)})
                    rows.append({'frame':k['frame'],'time_sec':k['time_sec'],'image':k['image'],'tokens':tokens})
                    total+=1
                results.append({'segment_id':s['segment_id'],'tracklet_id':t['tracklet_id'],'observations':rows})
                if total and total%30<6:print(json.dumps({'stage':'ocr','clip':cid,'crops_processed_this_run':total,'elapsed_sec':round(time.monotonic()-start,1)}),flush=True)
        save(target,{'clip_id':cid,'gallery_sha256':sha(path),'protocol_sha256':sha(protocol),'tracklets':results,
                     'human_verified':False,'identity_verified':False})
        print(json.dumps({'clip':cid,'tracklets':len(results),'crops_processed_this_run':total}),flush=True)


if __name__=='__main__':main()
