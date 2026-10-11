"""Independent Vision II camera timing pipeline; outputs are evidence candidates."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,yaml
from src.camera_timing_v2 import scan_video,SCAN_VERSION
cv2.setNumThreads(2)


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def metadata():
    manifest=yaml.safe_load((ROOT/'configs/penalty_slicing_manifest.yaml').read_text(encoding='utf-8'))
    return {f"{match}_p{p['number']:02d}_{p['kicker']}":{**p,'match':match,'source_video_filename':value['video_filename']}
            for match,value in manifest.items() for p in value['penalties']}


def scan(cid,meta,out,resume):
    path=out/'scan'/(cid+'.json');video=ROOT/'data/raw_clips'/(cid+'.mp4');digest=sha(video)
    if path.exists():
        if not resume:raise FileExistsError(path)
        existing=load(path)
        if existing['source_sha256']!=digest:raise ValueError('Cached source mismatch')
        if existing['boundary_method']!=SCAN_VERSION:raise ValueError('Detector version changed; use a new output directory')
        return existing
    result=scan_video(video);result.update(clip_id=cid,video_path=str(video),source_sha256=digest,manifest=meta,
        source_outcome_labels_read=False,human_verified=False,scan_code_sha256=sha(ROOT/'src/camera_timing_v2.py'))
    cap=cv2.VideoCapture(str(video));thumbs=out/'review/thumbs';thumbs.mkdir(parents=True,exist_ok=True)
    tiles=[]
    for segment in result['segments']:
        images=[]
        for fid in segment['sample_frame_ids']:
            cap.set(1,fid);ok,frame=cap.read()
            if not ok:raise ValueError('Cannot decode shot sample')
            small=cv2.resize(frame,(384,216));cv2.putText(small,f'f{fid} / {fid/result["fps"]:.2f}s',(8,205),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            images.append(small)
        while len(images)<3:images.append(images[-1].copy())
        strip=np.hstack(images);strip=cv2.copyMakeBorder(strip,28,0,0,0,cv2.BORDER_CONSTANT)
        label=f'{cid} | shot {segment["segment_id"]} | {segment["start_time_sec"]:.2f}-{segment["end_time_sec"]:.2f}s'
        cv2.putText(strip,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        name=f'{cid}_s{segment["segment_id"]:02d}.jpg';cv2.imwrite(str(thumbs/name),strip)
        segment['thumbnail']='thumbs/'+name;tiles.append(strip)
    cap.release()
    for i in range(0,len(tiles),6):
        cv2.imwrite(str(thumbs/f'{cid}_overview_{i//6+1}.jpg'),np.vstack(tiles[i:i+6]))
    if sha(video)!=digest:raise AssertionError('Source changed during scan')
    save(path,result);print(json.dumps({'stage':'scan','clip':cid,'segments':len(result['segments']),'frames':result['frame_count']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['scan','features','references','context','action','export','all'],default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/camera_timing_2026-10-10/release')
    parser.add_argument('--clips',nargs='+');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    manifest=metadata();ids=args.clips or list(manifest)
    out=args.output_dir.resolve();model=None
    if args.stage in ('features','action','all'):
        from scripts.camera_timing_features import load_model,extract_features,extract_action
        model=load_model()
    for cid in ids:
        if cid not in manifest:parser.error('Clip is not in the canonical manifest: '+cid)
        if args.stage in ('scan','all'):item=scan(cid,manifest[cid],out,args.resume)
        else:
            item=load(out/'scan'/f'{cid}.json')
            if sha(Path(item['video_path']))!=item['source_sha256']:raise ValueError('Source video hash mismatch')
        if args.stage in ('features','all'):
            result=extract_features(item,out,model,args.resume);save(out/'features'/f'{cid}.json',result)
            print(json.dumps({'stage':'features','clip':cid,'segments':len(result['segments'])}),flush=True)
        if args.stage=='references':
            from scripts.camera_timing_features import build_references
            path=out/'references_v2'/f'{cid}.json'
            if path.exists() and not args.resume:raise FileExistsError(path)
            if not path.exists():
                cap=cv2.VideoCapture(item['video_path']);refs=build_references(item,cap,None);cap.release();save(path,refs)
            print(json.dumps({'stage':'references','clip':cid,'keeper_refs':len(load(path)['keeper_histograms'])}),flush=True)
        if args.stage in ('context','all'):
            from scripts.camera_timing_context import extract
            save(out/'context_v2'/f'{cid}.json',extract(item,out,args.resume))
            print(json.dumps({'stage':'context','clip':cid}),flush=True)
        if args.stage in ('action','all'):
            features=load(out/'features'/f'{cid}.json');action=extract_action(item,features,model,out,args.resume)
            save(out/'action_v3'/f'{cid}.json',action)
            print(json.dumps({'stage':'action','clip':cid,'tested_segments':len(action['evidence'])}),flush=True)
    if args.stage in ('export','all'):
        from scripts.export_camera_timings import export
        export(out,ids)


if __name__=='__main__':main()
