"""Build target-blind within-shot jersey crop gallery from retained measurements."""
from pathlib import Path
import json,sys,argparse
from collections import Counter
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import read,save,sha,BASE,DEFAULT as PREVIOUS
from src.camera_timing_v2 import assign_kit_roles
from src.jersey_evidence import build_tracklets,torso_crop,select_keyframes
DEFAULT=ROOT/'outputs/jersey_retrieval_2026-10-11_v1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=DEFAULT);p.add_argument('--resume',action='store_true')
    args=p.parse_args();out=args.output_dir;out.mkdir(exist_ok=True,parents=True)
    cfg={'builder_sha256':sha(__file__),'algorithm_sha256':sha(ROOT/'src/jersey_evidence.py'),
         'source_baseline_sha256':sha(BASE/'camera_timings_candidates.json'),
         'sources':'all 197 baseline shots + previous 10Hz windows, not every frame of every shot',
         'min_person_height_px':160,'min_person_area_fraction':.025,'track_max_gap_sec':.4,'max_keyframes_per_tracklet':6,
         'frame_selection':'target-blind temporal bins and area/sharpness ranking; no OCR consulted',
         'human_labels_read':False,'outcome_labels_read':False,'independent_holdout':False}
    if (out/'gallery_protocol.json').exists():
        if read(out/'gallery_protocol.json')!=cfg:raise ValueError('Protocol changed')
    else:save(out/'gallery_protocol.json',cfg)
    pack=read(BASE/'camera_timings_candidates.json');all_clips=[];cv2.setNumThreads(2)
    for clip in pack['clips']:
        cid=clip['clip_id'];target=out/'gallery'/f'{cid}.json'
        if target.exists():
            if not args.resume:raise FileExistsError(target)
            record=read(target)
            if record['protocol_sha256']!=sha(out/'gallery_protocol.json'):raise ValueError('Cache changed')
            all_clips.append(record);continue
        feat_path=BASE/'features'/f'{cid}.json';dense_path=PREVIOUS/'measurements'/f'{cid}.json'
        features=read(feat_path);dense=read(dense_path);fps=clip['fps'];shots=[]
        video=Path(clip['raw_video_path'])
        if sha(video)!=clip['source_sha256']:raise ValueError('Raw video changed')
        cap=cv2.VideoCapture(str(video))
        for segment in clip['segments']:
            sid=segment['segment_id'];coarse=next(s for s in features['segments'] if s['segment_id']==sid)
            frames={s['frame']:{'frame':s['frame'],'people':assign_kit_roles(s['people'],features['references'])} for s in coarse['samples']}
            extra=next((s for s in dense['segments'] if s['segment_id']==sid),None)
            if extra:
                frames.update({f['frame']:{'frame':f['frame'],'people':f['people']} for f in extra['frames']})
            tracks=build_tracklets([frames[f] for f in sorted(frames)],fps)
            records=[]
            for track in tracks:
                if len(track['observations'])<2:continue
                candidates=[]
                for o in track['observations']:
                    cap.set(1,o['frame']);ok,image=cap.read()
                    if not ok:raise RuntimeError('Decode failed')
                    crop,info=torso_crop(image,o['person'])
                    if crop is not None:candidates.append({**o,'crop_info':info})
                selected=select_keyframes(candidates);keyframes=[]
                for o in selected:
                    cap.set(1,o['frame']);ok,image=cap.read()
                    if not ok:raise RuntimeError('Decode failed')
                    crop,info=torso_crop(image,o['person']);name=f'{cid}_s{sid:02}_t{track["tracklet_id"]:03}_f{o["frame"]:05}.jpg'
                    path=out/'crops'/name;path.parent.mkdir(exist_ok=True)
                    cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(path))
                    keyframes.append({'frame':o['frame'],'time_sec':o['time_sec'],'person_index':o['person_index'],
                        'image':'crops/'+name,'image_sha256':sha(path),'person_bbox':o['person']['bbox'],
                        'old_kit_role':o['person'].get('role'),'old_kit_distances':o['person'].get('appearance_distances'),**info})
                if keyframes:
                    records.append({'tracklet_id':track['tracklet_id'],'segment_id':sid,
                        'observations':[{'frame':o['frame'],'person_index':o['person_index'],'bbox':o['person']['bbox']} for o in track['observations']],
                        'start_frame':track['observations'][0]['frame'],'end_frame_exclusive':track['observations'][-1]['frame']+1,
                        'keyframes':keyframes,'identity_verified':False})
            shots.append({'segment_id':sid,'start_frame':segment['start_frame'],'end_frame_exclusive':segment['end_frame_exclusive'],
                          'available_pose_frames':len(frames),'tracklets':records})
        cap.release()
        result={'clip_id':cid,'fps':fps,'source_sha256':clip['source_sha256'],'protocol_sha256':sha(out/'gallery_protocol.json'),
                'source_feature_sha256':sha(feat_path),'source_dense_sha256':sha(dense_path),'shots':shots}
        save(target,result);all_clips.append(result)
        print(json.dumps({'clip':cid,'shots':len(shots),'tracklets':sum(len(s['tracklets']) for s in shots),
                          'keyframes':sum(len(t['keyframes']) for s in shots for t in s['tracklets'])}),flush=True)
    summary={'clips_indexed':len(all_clips),'shots_indexed':sum(len(c['shots']) for c in all_clips),
             'tracklets':sum(len(s['tracklets']) for c in all_clips for s in c['shots']),
             'keyframes':sum(len(t['keyframes']) for c in all_clips for s in c['shots'] for t in s['tracklets']),
             'human_ground_truth_records':0,'identity_verified':False}
    save(out/'gallery_summary.json',summary);print(summary)


if __name__=='__main__':main()
