"""Package a verified incremental experiment without changing earlier archives."""
from pathlib import Path
import sys,json,zipfile,hashlib,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_jersey_gallery import DEFAULT as OUT
from scripts.run_camera_partial_pose_experiment import read,save,sha
from scripts.finalize_jersey_experiment import SOURCE_FILES


def main():
    validation=read(OUT/'validation/summary.json')
    if not validation['tests']['targeted']['success']:raise ValueError('Tests did not pass')
    missing=[]
    for path in OUT.rglob('*.html'):
        for ref in re.findall(r'(?:href|src)="([^"]+)"',path.read_text(encoding='utf-8')):
            if ref.startswith(('https:','http:','#','data:')):continue
            if not (path.parent/ref).is_file():missing.append({'file':str(path),'reference':ref})
    if missing:raise ValueError(missing)
    save(OUT/'validation/html_links.json',{'missing_local_links':missing,'note':'Structural link check; not browser rendering QA.'})
    files=[p for p in OUT.rglob('*') if p.is_file()]
    files += [ROOT/n for n in SOURCE_FILES]+[ROOT/'scripts/package_jersey_experiment.py',ROOT/'requirements-jersey.txt']
    names=[p.relative_to(ROOT).as_posix() for p in files]
    if len(set(names))!=len(names):raise ValueError('Duplicate file names')
    manifest={'kind':'incremental_requires_existing_project_and_caches','human_ground_truth_included':False,
              'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
    archive=ROOT/'deliverables/VisionII_Jersey_Retrieval_2026-10-11_Experimental.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('START_HERE.txt','Open outputs/jersey_retrieval_2026-10-11_v1/report.html and README.txt.\nIncremental experiment only; no raw-video corpus, binary runtime, model weights or human ground truth.\nUse camera_integration_v2 plus review_candidates.json; evaluation_v1 hard OCR veto is abandoned.\n')
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip();checks=[hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in manifest['files']]
        if bad or not all(checks):raise ValueError('Archive integrity failure')
    result={'zip':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'crc_bad_entry':bad,
            'manifest_files':len(checks),'all_entry_hashes_match':all(checks),'example_videos_fully_decoded':2,'missing_report_links':0}
    save(ROOT/'deliverables/VisionII_Jersey_Retrieval_2026-10-11_Experimental_package_check.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
