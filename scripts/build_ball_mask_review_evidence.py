"""Render automatic review points; they are not ball labels or inference inputs."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from scripts.build_ball_detection_seed_evidence import source_frame, evidence_tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    out = args.directory
    rows = json.loads((out/'summary.json').read_text(encoding='utf-8'))['rows']
    checks = []
    for rr in rows:
        cid = rr['clip_id']
        r = json.loads((out/'results'/f'{cid}.json').read_text(encoding='utf-8'))
        cache = json.loads((out/'inputs'/f'{cid}.json').read_text(encoding='utf-8'))
        v = r['variants']['mask_and_radius_35']
        seed = v.get('effective_stationary_seed', v['seed'])
        if not seed:
            picks = []
        else:
            origin = np.median([q['stabilized_xy'] for q in seed], axis=0)
            goal = np.array(cache['goal_center'])
            reach = np.linalg.norm(goal-origin)
            ds = [q for q in v['trajectory'] if .15*reach < np.linalg.norm(np.array(q['stabilized_xy'])-origin) < .7*reach]
            picks = [max(seed, key=lambda q:q['confidence'])]
            if ds:
                picks += [ds[0]]
                if ds[-1]['detection_id'] != ds[0]['detection_id']: picks += [ds[-1]]
        selected=[]
        panel=Image.new('RGB',(900,270),(25,29,32))
        for i,d in enumerate(picks):
            im=source_frame(r['source_video_path'],d['frame'])
            panel.paste(evidence_tile(im,d,cid),(300*i,0))
            selected.append({'frame':d['frame'],'detection_id':d['detection_id'],
              'xy':[d['x'],d['y']],'sources':d['sources']})
        panel.save(out/'media'/f'{cid}_midflight_check.jpg')
        checks.append({'clip_id':cid,'selected_points':selected,'human_verified':False,
          'selection_policy':'strongest effective seed and first-last mid-flight candidates; identity unverified'})
    for j in range(0,len(rows),6):
        chunk=rows[j:j+6]
        mid=Image.new('RGB',(900,270*len(chunk)),(25,29,32))
        flight=Image.new('RGB',(1200,250*len(chunk)),(25,29,32))
        for n,r in enumerate(chunk):
            cid=r['clip_id']
            mid.paste(Image.open(out/'media'/f'{cid}_midflight_check.jpg'),(0,270*n))
            for i,f in enumerate(r['sample_frames'][1:5]):
                im=Image.open(out/'media'/f'{cid}_f{f}.jpg')
                sub=Image.new('RGB',(300,250),(25,29,32))
                sub.paste(im.crop((650,40,830,260)),(50,27))
                ImageDraw.Draw(sub).text((8,7),f'{cid} frame{f}',fill='white')
                flight.paste(sub,(i*300,n*250))
        mid.save(out/f'midflight_grid_{j//6+1}.jpg')
        flight.save(out/f'flight_grid_{j//6+1}.jpg')
    (out/'midflight_check_points.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Review panels: {len(checks)}; labels not generated')


if __name__=='__main__':main()
