"""Render separately recorded AI visual samples; never used by tracking."""
from pathlib import Path
import argparse
import json
from PIL import Image
from scripts.build_ball_detection_seed_evidence import source_frame, evidence_tile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();out=args.directory
    rows=json.loads((out/'visual_review.json').read_text(encoding='utf-8'))
    for row in rows:
        cid=row['clip_id'];r=json.loads((out/'results'/f'{cid}.json').read_text(encoding='utf-8'))
        v=r['variants']['mask_and_radius_35']
        allowed={q['detection_id']:q for q in v['trajectory']+v.get('effective_stationary_seed',v['seed'])}
        picks=[row['seed_evidence']]+row['flight_evidence'][:2]
        panel=Image.new('RGB',(900,270),(25,29,32))
        for i,entry in enumerate(picks):
            d=allowed[entry['detection_id']];assert d['frame']==entry['frame']
            panel.paste(evidence_tile(source_frame(r['source_video_path'],d['frame']),d,cid),(300*i,0))
        panel.save(out/'media'/f'{cid}_visual_review_sample.jpg')
    print(f'{len(rows)} visual sample panels; AI review, no human ground truth')


if __name__=='__main__':main()
