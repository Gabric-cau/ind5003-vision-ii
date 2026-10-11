"""Archive the current incremental direction correction and its evidence."""
from pathlib import Path
import sys,json,zipfile,hashlib,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha
OUT=ROOT/'outputs/goalkeeper_direction_2026-10-11'


def main():
    check=read(OUT/'validation.json')
    if not check['tests_passed'] or not check['source_and_old_results_unchanged']:raise ValueError('Validation failed')
    report=(OUT/'report.html').read_text(encoding='utf-8')
    for link in re.findall(r'(?:href|src)="([^"]+)"',report):
        if not (OUT/link).is_file():raise ValueError(f'Missing report file: {link}')
    source_names=['src/goalkeeper_observation_guard.py','src/goalkeeper_direction_evidence.py',
        'scripts/run_goalkeeper_direction_update.py','scripts/review_goalkeeper_direction_update.py',
        'scripts/package_goalkeeper_direction_update.py','scripts/run_evening_gk_observation_guard.py',
        'tests/test_goalkeeper_direction_evidence.py','tests/test_goalkeeper_observation_guard.py',
        'tests/test_goalkeeper_analysis.py','tests/test_goalkeeper_bounded.py','tests/test_goalkeeper_qa_gap.py']
    files=sorted([p for p in OUT.rglob('*') if p.is_file()]+[ROOT/n for n in source_names])
    manifest={'kind':'incremental_requires_existing_project_and_cached_tracks',
        'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    archive=ROOT/'deliverables/VisionII_Goalkeeper_Direction_Update_2026-10-11.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('START_HERE.txt','Open outputs/goalkeeper_direction_2026-10-11/report.html and README.txt.\nIncremental direction correction: cached-track evidence, not labelled accuracy.\nThe report video references the earlier full project deliverable; source images and JSON are included.\n')
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip();valid=all(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in manifest['files'])
        if bad or not valid:raise ValueError('Archive integrity failed')
    result={'zip':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'manifest_files':len(files),'crc_bad_entry':bad,'all_entry_hashes_match':valid}
    save(ROOT/'deliverables/VisionII_Goalkeeper_Direction_Update_2026-10-11_package_check.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
