"""Create a local incremental experiment archive and two decodable examples."""
from pathlib import Path
import sys,json,subprocess,zipfile,hashlib,html,importlib.metadata
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import DEFAULT,read,save,sha


def main():
    out=DEFAULT;examples=out/'examples';examples.mkdir(exist_ok=True)
    encoder=ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    rows=read(out/'review_candidates.json')['candidates_without_known_ai_flags'];examples_log=[]
    for cid in ('col_eng_p02_kane_eng9','col_eng_p03_cuadrado_col11'):
        row=next(r for r in rows if r['clip_id']==cid);target=examples/f'{cid}_candidate_interval.mp4'
        if target.exists():raise FileExistsError(target)
        cmd=[str(encoder),'-hide_banner','-loglevel','error','-n','-ss',str(row['start_time_sec']),
             '-i',str(ROOT/'data/raw_clips'/f'{cid}.mp4'),'-t',str(row['end_time_sec']-row['start_time_sec']),
             '-map','0:v:0','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(target)]
        subprocess.run(cmd,check=True,capture_output=True)
        cap=cv2.VideoCapture(str(target));reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded+=1
        cap.release()
        if decoded==0 or decoded!=reported:raise ValueError('Incomplete output decode')
        examples_log.append({'clip_id':cid,'start_sec':row['start_time_sec'],'end_sec':row['end_time_sec'],
             'video_file':target.name,'source_sha256':sha(ROOT/'data/raw_clips'/f'{cid}.mp4'),
             'output_sha256':sha(target),'frames_reported':reported,'frames_decoded':decoded,
             'ffmpeg_argv':cmd,'playback_speed_verified':False})
    save(examples/'decode_validation.json',examples_log)
    qa=out/'visual_qa';images=sorted(qa.glob('*.jpg'))
    html_qa='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AI诊断图</title><style>body{max-width:1500px;margin:auto;font:18px/1.7 system-ui}img{width:100%;height:auto}p{padding:18px;background:#fff0ce}</style><h1>54段候选的静态证据</h1><p>每段3帧。黄色框=颜色匹配的主罚者候选，蓝色框=守门员颜色候选，绿色=球门几何候选；这些框可能误检。此页经过AI查看，不是人工真值。请结合 <a href="ai_findings.json">AI发现记录</a> 理解错误。</p>'
    html_qa+=''.join(f'<h2>{p.stem}</h2><img src="{p.name}" alt="{p.stem}" loading="lazy">' for p in images)+'</html>'
    (qa/'index.html').write_text(html_qa,encoding='utf-8')
    report=(out/'report.html').read_text(encoding='utf-8')
    cards='<h2>两个恢复区间的原画面截取</h2><p>以下视频只显示候选区间，不代表完整动作或慢动作验证。无生成帧；H.264转码并逐帧解码检查。</p><div class="cards">'
    for r in examples_log:
        cards+=f'<div class="card"><p>{r["clip_id"]}<br>原片 {r["start_sec"]:.2f}–{r["end_sec"]:.2f} 秒</p><video controls preload="metadata" src="examples/{r["video_file"]}"></video></div>'
    cards+='</div><p><a href="visual_qa/index.html">打开全部54段采样图与AI检查记录</a></p>'
    report=report.replace('<h2>二、哪些实验没有达到目标</h2>',cards+'<h2>二、哪些实验没有达到目标</h2>')
    (out/'report.html').write_text(report,encoding='utf-8')
    versions={'python':sys.version,'executable':sys.executable}
    for name in ('numpy','opencv-python','torch','ultralytics','pytest'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not visible in interpreter metadata; project .venv may be appended by model loader'
    save(out/'validation/environment_versions.json',versions)
    files=list(out.rglob('*'));files=[p for p in files if p.is_file()]
    source_names=('src/camera_partial_alignment.py','src/camera_view_intervals.py','src/camera_view_guard.py',
        'src/goal_temporal_tracking.py','scripts/run_camera_partial_pose_experiment.py',
        'scripts/evaluate_camera_partial_alignment.py','scripts/run_camera_dense_view_experiment.py',
        'scripts/render_camera_interval_qa.py','scripts/run_goal_temporal_pilot.py','scripts/run_camera_guarded_views.py',
        'scripts/finalize_camera_optimization.py','scripts/package_camera_optimization.py',
        'tests/test_camera_partial_alignment.py','tests/test_camera_view_intervals.py',
        'tests/test_camera_view_guard.py','tests/test_goal_temporal_tracking.py')
    files += [ROOT/n for n in source_names]
    if (ROOT/'requirements.txt').exists():files.append(ROOT/'requirements.txt')
    manifest={'package_kind':'incremental_experiment_requires_existing_project','files':[
        {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
    archive=ROOT/'deliverables/VisionII_Optimization_2026-10-11_Experimental.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('START_HERE.txt','Open outputs/camera_partial_alignment_2026-10-11/report.html and README.txt.\nExperimental candidates, not verified results. This is an incremental package, not a standalone project.\n')
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        checks=[hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in manifest['files']]
        if bad or not all(checks):raise ValueError('Archive validation failed')
    save(ROOT/'deliverables/VisionII_Optimization_2026-10-11_Experimental_package_check.json',
         {'zip':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'crc_bad_entry':bad,
          'manifest_entries':len(checks),'all_entry_hashes_match':all(checks),'example_videos_fully_decoded':len(examples_log)})
    print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,'manifest_entries':len(checks),'crc_bad_entry':bad},indent=2))


if __name__=='__main__':main()
