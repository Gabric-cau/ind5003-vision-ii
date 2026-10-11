"""Read-only project audit. Never runs extraction or changes analysis outputs."""
from pathlib import Path
import csv
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = {"date": "2026-10-09", "csv": {}, "notebooks": {}, "json": {}}
    for p in sorted((ROOT / "data").glob("*/*.csv")):
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        if not rows:
            continue
        result["csv"][str(p.relative_to(ROOT))] = {
            "rows": len(rows), "columns": list(rows[0]), "first": rows[0],
            "nonempty": {k: sum(r[k].strip().lower() not in {"", "nan", "none", "null"} for r in rows) for k in rows[0]},
            "categories": {k: dict(Counter(r[k] for r in rows)) for k in rows[0]
                           if len(set(r[k] for r in rows)) <= 12},
        }
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(p.read_text(encoding="utf-8"))
        result["notebooks"][p.name] = [{
            "cell": i, "type": c["cell_type"], "execution_count": c.get("execution_count"),
            "source": "".join(c.get("source", [])),
            "output_types": [o.get("output_type") for o in c.get("outputs", [])],
            "output_text": ["".join(o.get("text", []))[:1800] for o in c.get("outputs", []) if o.get("text")],
            "errors": [o.get("evalue") for o in c.get("outputs", []) if o.get("output_type") == "error"]
        } for i, c in enumerate(nb["cells"])]
    targets = [
        "outputs/style_aware_steps_2026-10-09/classified/summary.json",
        "outputs/style_aware_steps_2026-10-09/validation.json",
        "outputs/style_aware_steps_2026-10-09/model_comparison.json",
        "outputs/phase_optimization_2026-10-09/classified/summary.json",
        "outputs/terminal_footwork_2026-10-09/classified/summary.json",
        "outputs/mask_optimization_2026-10-08/summary.json",
    ]
    for name in targets:
        p = ROOT / name
        if p.exists():
            result["json"][name] = json.loads(p.read_text(encoding="utf-8-sig"))
    dest = ROOT / "outputs/completion_audit_2026-10-09"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, info in result["csv"].items():
        print(name, "rows:", info["rows"], "nonempty:", info["nonempty"])
        print("categories:", info["categories"])
    print("notebooks", {n: len(c) for n, c in result["notebooks"].items()})
    print("saved", dest / "evidence.json")


if __name__ == "__main__":
    main()
