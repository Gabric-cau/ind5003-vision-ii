"""Safe cached-evidence entry point for the evening experiment.

validate: selected tests/replays/video hashes (writes validation.json).
bundle: descriptive canonical join/plots/current-source snapshot.
inspect: compact read-only metrics, no expensive inference.
Heavy detector runs deliberately remain explicit separate experiments.
"""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["inspect","validate","bundle"])
    args=parser.parse_args()
    if args.mode=="inspect":
        path=ROOT/"outputs/optimization_evening_2026-10-09/integration/metrics.json"
        metrics=json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({k:v for k,v in metrics.items() if k not in ("new_count_rows","tests")},ensure_ascii=False,indent=2))
        return 0
    script={"validate":"validate_evening_experiments.py","bundle":"build_evening_result_bundle.py"}[args.mode]
    return subprocess.call([sys.executable,"-X","utf8",str(ROOT/"scripts"/script)],cwd=ROOT)


if __name__=="__main__":raise SystemExit(main())
