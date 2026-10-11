"""Browser-compatible review copies; never touch original clips."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.runtime import resolve_ffmpeg, file_sha256
from src.project_paths import resolve_project_video


def sha(p):return file_sha256(p)
def prepare(out,resume=False,ffmpeg=None,source_dir=None):
    ffmpeg=resolve_ffmpeg(ffmpeg,ROOT)
    folder=out/'review/videos';folder.mkdir(parents=True,exist_ok=True);records=[]
    for p in sorted((out/'scan').glob('*.json')):
        scan=json.loads(p.read_text(encoding='utf-8'))
        source=resolve_project_video(ROOT,scan['video_path'],scan['clip_id'],source_dir);before=sha(source)
        if before!=scan['source_sha256']:raise ValueError('Source mismatch')
        dest=folder/(scan['clip_id']+'.mp4');qa=folder/(scan['clip_id']+'.qa.json')
        if dest.exists() and resume and qa.exists():
            record=json.loads(qa.read_text(encoding='utf-8'))
            if record['source_sha256']!=before or record['output_sha256']!=sha(dest):raise ValueError('Review cache mismatch')
            records.append(record);continue
        if dest.exists():raise FileExistsError(dest)
        command=[ffmpeg,'-hide_banner','-loglevel','error','-n','-threads','2','-i',str(source),'-map','0:v:0','-map','0:a?',
                 '-vf','scale=1280:-2','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-threads','2',
                 '-c:a','aac','-b:a','128k','-movflags','+faststart',str(dest)]
        subprocess.run(command,check=True)
        cap=cv2.VideoCapture(str(dest));count=0;fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release()
        if count!=scan['frame_count'] or abs(fps-scan['fps'])>1e-5:raise AssertionError('Review copy timing mismatch')
        if sha(source)!=before:raise AssertionError('Original changed')
        # Decoder scans the entire compressed stream, including audio if present.
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-xerror','-i',str(dest),'-f','null','-'],check=True)
        record={'clip_id':scan['clip_id'],'source_sha256':before,'output_sha256':sha(dest),'decoded_frames':count,'fps':fps,
                'codec_requested':'libx264/yuv420p/faststart','ffmpeg':ffmpeg,
                'full_stream_decode_pass':True,'original_unchanged':True}
        qa.write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
        print(json.dumps(record),flush=True)
    (out/'media_audit.json').write_text(json.dumps({'clips':len(records),'records':records},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/camera_timing_2026-10-10/release')
    p.add_argument('--resume',action='store_true');p.add_argument('--ffmpeg')
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/raw_clips')
    a=p.parse_args();prepare(a.output_dir.resolve(),a.resume,a.ffmpeg,a.source_dir)
