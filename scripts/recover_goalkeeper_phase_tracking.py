"""Explicit expanded-ROI ablation; keep physical/body pixel scale unchanged."""
from pathlib import Path
import argparse, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_goalkeeper_height_phase as pipeline
from scripts.run_goalkeeper_direction_update import save,sha,read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--roi-expansion',type=float,default=1.6)
    args,remaining=parser.parse_known_args()
    if not 1 < args.roi_expansion <= 2.5:raise ValueError('Bounded ROI expansion expected')
    if '--cache-run' in remaining:raise ValueError('Changed ROI cannot reuse old crop masks')
    out=Path(remaining[remaining.index('--output-dir')+1]).resolve()
    out.mkdir(parents=True,exist_ok=True)
    recovery=dict(policy='expanded_static_roi_ablation_v1',roi_expansion=args.roi_expansion,
                  anatomical_body_scale_unchanged=True,source_sha256=sha(Path(__file__)),
                  no_user_labels_read_for_inference=True)
    p=out/'recovery_protocol.json'
    if p.exists() and read(p)!=recovery:raise ValueError('Recovery protocol changed')
    save(p,recovery)
    normal=pipeline.extend
    def widened(data,kick,seconds):
        original_scale=data['scale']
        data['scale']=original_scale*args.roi_expansion
        try:result=normal(data,kick,seconds)
        finally:data['scale']=original_scale
        return result
    pipeline.extend=widened
    sys.argv=[sys.argv[0],*remaining]
    pipeline.main()


if __name__=='__main__':main()
