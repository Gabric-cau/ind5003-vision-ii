"""Build short original-pixel event panels without loading outcome labels."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime,timezone
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw
from scripts.build_ball_detection_seed_evidence import source_frame


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    (out/'media').mkdir(parents=True);(out/'source_snapshot').mkdir()
    parent=ROOT/'outputs/ball_shot_focus_2026-10-11';summary=load(parent/'summary.json')
    rows=[]
    for i,r in enumerate(summary['rows'],1):
        cid=r['clip_id'];cache=ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json'
        c=load(cache);assert sha(c['video_path'])==c['source_sha256']
        boundary=r['phase']['main_end_frame']
        selected=sorted(set(max(c['start_frame'],min(c['end_frame_exclusive']-1,boundary+offset))
                             for offset in [-4,-1,2,5,8,11]))
        panel=Image.new('RGB',(1200,1200),(25,29,32));evidence=[]
        for j,f in enumerate(selected):
            im=source_frame(c['video_path'],f)
            q=c['goal_proposals'].get(str(f),{}).get('selected')
            if q:
                x0,y0,x1,y1=q['bbox'];w=x1-x0;h=y1-y0
                box=(max(0,int(x0-.45*w)),max(0,int(y0-.20*h)),
                     min(im.width,int(x1+.45*w)),min(im.height,int(y1+.55*h)))
            else:box=tuple(c['goal_roi'])
            crop=im.crop(box);crop.thumbnail((590,365),Image.Resampling.LANCZOS)
            tile=Image.new('RGB',(600,400),(25,29,32))
            tile.paste(crop,((600-crop.width)//2,31))
            ImageDraw.Draw(tile).text((10,7),f'Event {i:02d} | original source frame {f}',fill='white')
            panel.paste(tile,(j%2*600,j//2*400))
            evidence.append({'frame':f,'crop_bbox':list(box),'has_goal_proposal':q is not None})
        panel.save(out/'media'/f'event_{i:02d}.jpg',quality=95)
        rows.append({'event_id':i,'clip_id':cid,'panel':f'event_{i:02d}.jpg',
          'frames':evidence,'source_video_path':c['video_path'],'source_sha256':c['source_sha256'],
          'source_cache_sha256':sha(cache),'automatic_outcome':r['outcome'],
          'automatic_crossing_established':r['phase']['whole_ball_crossing_established'],
          'automatic_no_goal_hint':r['phase']['no_goal_hint'],'visual_assessment':'pending',
          'human_verified':False,'accuracy_validated':False})
        print(f'Event {i:02d}: frames {[q["frame"] for q in evidence]}',flush=True)
    save=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    save(out/'events.json',rows)
    save(out/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
      'clips':39,'parent_summary_sha256':sha(parent/'summary.json'),
      'outcome_labels_used':False,'visual_review_used_for_frame_selection':False,
      'new_neural_inference':False,'crop_policy':'automatic goal bbox, no ball coordinate labels',
      'sampling_policy':'automatic focus boundary -4,-1,+2,+5,+8,+11 original source frames',
      'panel_labels':'event ordinal only, no player name or expected outcome',
      'code_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
      'human_verified':False,'accuracy_validated':False})
    (out/'source_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
