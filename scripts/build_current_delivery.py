"""Package current usable engineering artifacts, never promote candidates to truth."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVENING = ROOT / 'outputs/optimization_evening_2026-10-09'
CAMERA = ROOT / 'outputs/camera_timing_2026-10-10/release'
RETEST = ROOT / 'outputs/camera_timing_retest_2026-10-10_01'
sys.path.insert(0, str(ROOT))
from src.style_aware_steps import analyze_style_aware_steps


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


CSS = '''body{max-width:1160px;margin:32px auto;padding:0 22px 60px;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#16343b;background:#f4f7f6}h1{font-size:30px}h2{margin-top:30px}a{color:#146d70}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border:1px solid #d8e2e2;font-size:14px}th{background:#deeeee}code{word-break:break-all}pre{white-space:pre-wrap;background:white;padding:18px}.notice{padding:18px;border-left:5px solid #d4a646;background:#fff5d8}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.card{padding:18px;background:white;border:1px solid #d8e2e2;border-radius:10px}.muted{color:#587078}video{width:100%;max-height:620px;background:#000} .scroll{overflow-x:auto}'''


def page(title, body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + escape(title) + '</title><style>' + CSS + '</style><main>' + body + '</main></html>'


VERIFY = '''"""Offline checks: package hashes, candidate invariants and 39 cached style replays.
No model downloads, no inference, no modification of any result or label.
"""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
errors=[]
for item in manifest['files']:
    p=(root/item['path']).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        errors.append('missing/unsafe: '+item['path']);continue
    with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=item['sha256'] or p.stat().st_size!=item['bytes']:errors.append('hash/size: '+item['path'])
sys.path.insert(0,str(root/'03_source'))
from src.style_aware_steps import analyze_style_aware_steps,StepStyleConfig
rows=json.loads((root/'01_results/steps_candidates_39.json').read_text(encoding='utf-8'))['rows']
for row in rows:
    r=json.loads((root/'01_results'/row['evidence_file']).read_text(encoding='utf-8'))
    a=r['step_analysis']
    if analyze_style_aware_steps(r,StepStyleConfig(**a['thresholds']))!=a:errors.append('style replay: '+row['clip_id'])
    if sum(v or 0 for v in a['count_partition'].values())!=a['candidate_step_count']:errors.append('partition: '+row['clip_id'])
    if a['verified_step_count'] is not None or a['human_verified'] or a['accuracy_validated']:errors.append('label provenance: '+row['clip_id'])
camera=json.loads((root/'01_results/camera/camera_timings_candidates.json').read_text(encoding='utf-8'))
if len(rows)!=39 or len(camera['clips'])!=39:errors.append('canonical clip count')
for c in camera['clips']:
    end=0
    for s in c['segments']:
        if s['start_frame']!=end or s['end_frame_exclusive']<=end or s['human_verified']:errors.append('camera invariant: '+c['clip_id'])
        end=s['end_frame_exclusive']
    if end!=c['frame_count']:errors.append('camera duration: '+c['clip_id'])
print(json.dumps({'passed':not errors,'files_checked':len(manifest['files']),'style_replays':len(rows),'camera_clips':len(camera['clips']),'errors':errors,'accuracy_validated':False},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
'''


def build(destination):
    if destination.exists():
        raise FileExistsError('Will not overwrite an existing delivery: ' + str(destination))
    destination.mkdir(parents=True)
    provenance = {}

    def copy(source, target):
        p = destination / target
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, p)
        digest = sha(source)
        assert sha(p) == digest
        provenance[p.relative_to(destination).as_posix()] = {'original_path': str(source), 'original_sha256': digest}

    def text(target, value):
        p = destination / target
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value, encoding='utf-8')

    canonical = read(EVENING / 'integration/canonical_results_39.json')['rows']
    camera = read(RETEST / 'camera_timings_candidates.json')
    retest = read(RETEST / 'retest_report.json')
    assert camera == read(CAMERA / 'camera_timings_candidates.json')
    assert len(canonical) == len(camera['clips']) == 39
    assert [r['clip_id'] for r in canonical] == [c['clip_id'] for c in camera['clips']]
    assert retest['original_unchanged_count'] == retest['media_decode_pass_count'] == 39
    media_old = {r['clip_id']: r for r in read(CAMERA / 'media_audit.json')['records']}
    steps, keepers, videos, table = [], [], [], []
    styles = {'continuous_approach_candidate':'连续助跑候选', 'terminal_adjustment_candidate':'末段调整候选', 'undetermined':'方式未定'}
    directions = {'image_left':'画面左侧', 'image_right':'画面右侧', 'center':'居中', 'unknown':'未知'}
    source_checks = []

    for row, clip in zip(canonical, camera['clips'], strict=True):
        cid = row['clip_id']
        raw = ROOT / 'data/raw_clips' / (cid + '.mp4')
        assert sha(raw) == clip['source_sha256']
        video = CAMERA / 'review/videos' / (cid + '.mp4')
        assert sha(video) == media_old[cid]['output_sha256']
        copy(video, '02_video_review/camera/videos/' + video.name)
        videos.append({'clip_id':cid, 'original_video_path':str(raw), 'original_sha256':clip['source_sha256'],
                       'playback_file':'02_video_review/camera/videos/' + video.name, 'playback_sha256':sha(video),
                       'fps':clip['fps'], 'frame_count':clip['frame_count'], 'duration_sec':clip['duration_sec'],
                       'is_browser_compatible_reencode':True, 'source_match':clip['source_match_video'],
                       'manifest_offset_sec':clip['manifest_offset_sec'], 'offset_independently_verified':False})
        step_file = EVENING / 'steps_v4/classified' / (cid + '.json')
        r = read(step_file)
        assert analyze_style_aware_steps(r) == r['step_analysis']
        a = r['step_analysis']
        assert a['verified_step_count'] is None and not a['human_verified'] and not a['accuracy_validated']
        assert a['candidate_step_count'] == row['steps']['v4_candidate_count']
        assert sum(v or 0 for v in a['count_partition'].values()) == a['candidate_step_count']
        evidence = 'steps_raw/' + cid + '.json'
        copy(step_file, '01_results/' + evidence)
        step = {'clip_id':cid, 'match':row['match_title'], 'kicker':row['kicker'],
                'candidate_step_count':a['candidate_step_count'], 'attempt_style':a['attempt_style'],
                'count_partition':a['count_partition'], 'parameter_range':row['steps']['parameter_range'],
                'partial_approach_known':row['steps']['partial_approach_known'],
                'contact_candidate':row['contact'], 'evidence_file':evidence,
                'verified_step_count':None, 'human_verified':False, 'accuracy_validated':False, 'review_required':True}
        steps.append(step)
        keeper_file = EVENING / 'goal_observation_guard' / (cid + '.json')
        gk = read(keeper_file)
        accepted = row['goalkeeper']['guarded_qa_accepted']
        assert gk['auto_accepted'] == accepted
        if accepted:
            copy(keeper_file, '01_results/goalkeeper_raw/' + cid + '.json')
            keepers.append({'clip_id':cid, 'qa_rule_accepted':True, 'identity_verified':False,
                            'pre_kick_movement_candidate':gk['gk_prekick_movement_level'],
                            'dive_direction_candidate':gk['dive_direction'], 'dive_height_candidate':gk['dive_height'],
                            'observation_window':row['goalkeeper']['observability'],
                            'evidence_file':'goalkeeper_raw/' + cid + '.json',
                            'human_verified':False, 'accuracy_validated':False, 'review_required':True})
        for name, expected in r.get('terminal_provenance', {}).get('code_sha256', {}).items():
            source_checks.append({'clip_id':cid, 'path':name, 'hash_matches':sha(ROOT / name) == expected})
        partial = '仅可见后段；非完整助跑' if step['partial_approach_known'] else '未证实完整覆盖，不代表缺失一定不存在'
        gk_text = directions.get(gk['dive_direction'], gk['dive_direction']) + ' / ' + gk['dive_height'] + '（候选）' if accepted else '未列入可用子集；待核'
        table.append(f'<tr><td><a href="02_video_review/clips/{cid}.html">{escape(cid)}</a></td><td>{a["candidate_step_count"]}</td><td>{styles[a["attempt_style"]]}</td><td>{step["parameter_range"]}</td><td>{escape(gk_text)}</td><td>{"仅后段" if step["partial_approach_known"] else "待确认"}</td></tr>')
        detail = f'''<a href="../../index.html">← 成果汇总</a><h1>{escape(cid)}</h1>
<p class="notice">所有数字和分类都是自动候选，不是人工确认；参数范围不是置信区间。</p>
<video controls preload="metadata" src="../camera/videos/{cid}.mp4"></video>
<div class="cards"><div class="card">助跑步数候选<h2>{a['candidate_step_count']}</h2>{styles[a['attempt_style']]}</div><div class="card">参数范围<h2>{step['parameter_range']}</h2>{partial}</div><div class="card">门将<h2>{escape(gk_text)}</h2>方向是画面方向；不是守门员自身左右。</div></div>
<h2>计数分区</h2><pre>{escape(json.dumps(a['count_partition'],ensure_ascii=False,indent=2))}</pre>
<p>contact_guard：触球保护区候选，保留在总数内；null：未估计，不是 0。步数是姿态周期候选，不等于已确认的落脚。</p>
<p>名义触球锚点：第 {row['contact']['contact_frame']} 帧（零起算）；{row['contact']['contact_time_sec']:.3f} 秒，仍需复核。</p>
<p><a href="../../01_results/{evidence}">完整步数证据 JSON</a> · <a href="../camera/index.html">机位逐帧复核工具（在选择框中选择本视频）</a></p>'''
        text('02_video_review/clips/' + cid + '.html', page(cid, detail))

    assert len(keepers) == 15
    assert all(c['hash_matches'] for c in source_checks)
    dump(destination / '01_results/steps_candidates_39.json', {'status':'usable_candidate_outputs_not_ground_truth','rows':steps})
    dump(destination / '01_results/goalkeeper_qa_accepted_15.json', {
        'status':'rule_accepted_subset_not_validated_accuracy', 'selection_rule':'guarded_qa_accepted == true',
        'note':'The unmodified evidence may say review_required=false; that is the old rule-QA flag, NOT human verification. Delivery review_required remains true.',
        'all_clips':39,'subset_clips':len(keepers),'rows':keepers})
    dump(destination / '01_results/video_index_39.json', {'clips':videos})
    dump(destination / '06_pending/goalkeeper_selection_status_39.json', {
        'rows':[{'clip_id':r['clip_id'],'included_in_rule_accepted_subset':r['goalkeeper']['guarded_qa_accepted'],
                 'human_verified':False,'review_required':True} for r in canonical]})
    copy(ROOT / 'configs/penalty_slicing_manifest.yaml', '01_results/penalty_slicing_manifest.yaml')
    for name in ('camera_timings_candidates.json','camera_timings_candidates.csv','summary.json'):
        copy(RETEST / name, '01_results/camera/' + name)
    for name in ('index.html','app.js','style.css','data.js','camera_timings_candidates.csv'):
        copy(CAMERA / 'review' / name, '02_video_review/camera/' + name)
    for p in sorted((CAMERA / 'review/thumbs').glob('*.jpg')):
        copy(p, '02_video_review/camera/thumbs/' + p.name)

    for folder in ('src','scripts','tests','assets','configs'):
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.js','.html','.css','.yaml'):
                copy(p, '03_source/' + p.relative_to(ROOT).as_posix())
    for name in ('retest_report.html','retest_report.json','retest_regression.json','retest_tests.json',
                 'retest_browser.json','retest_browser_screenshot.jpg','targeted.log','full_unittest_discovery.log'):
        copy(RETEST / name, '04_validation/' + name)
    copy(EVENING / 'integration/environment.json', '04_validation/runtime_observed_2026-10-09.json')
    copy(EVENING / 'validation.json', '04_validation/historical_step_validation_2026-10-09.json')
    copy(ROOT / 'scripts/serve_penalty_phase_review.py', 'serve_review.py')
    text('verify_delivery.py', VERIFY)
    text('start_review.ps1', '''param([string]$PythonExe = "python", [int]$Port = 8885)
$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "serve_review.py"
Write-Host "Open http://127.0.0.1:$Port/index.html after the server starts. Ctrl+C stops it."
& $PythonExe -X utf8 $entry --directory $PSScriptRoot --port $Port
''')
    copy(ROOT / 'requirements-evening-experiment.txt', 'requirements.txt')
    copy(ROOT / 'requirements-camera.txt', '03_source/requirements-camera.txt')
    text('requirements-tests-optional.txt', '# Only needed to collect pytest-function tests. Not installed/tested in the recorded runtime.\npytest>=8,<10\n')
    styles_count = dict(Counter(r['attempt_style'] for r in steps))
    timestamp = datetime.now(timezone.utc).isoformat()
    delivery = {'created_at_utc':timestamp, 'date_singapore':'2026-10-10', 'canonical_clips':39,
                'step_candidate_rows':len(steps), 'step_styles':styles_count, 'keeper_rule_accepted_rows':len(keepers),
                'camera_candidate_segments':camera['summary']['segments'], 'human_ground_truth_records':0,
                'accuracy_validated':False, 'source_code_hash_checks':source_checks,
                'scope':'Curated readable/playable candidate evidence and source snapshot; not a standalone raw-to-all-features training/inference release.',
                'originals_not_modified':True, 'video_copies':'Existing verified H264 transcodes, NOT bit-identical original raw clips.'}
    dump(destination / '04_validation/delivery_build_checks.json', delivery)
    pending = '''# 尚未列为完成的部分

这些项目仍保留在原工程中，本包不把它们包装成可靠最终答案：

- 左右脚：固定锚点与窗口敏感性仅 1/39 满足既有稳定性规则；稳定性本身也不是正确率。
- 真实俯视助跑轨迹：没有已验证的地面单应性；球门竖直平面变换不能替代它。
- 门将：仅 15/39 通过当前完整观察窗口及规则 QA；本包保留这 15 条候选，其他 24 条未列入子集。
- 球轨迹、真实终点和自动 goal/no-goal：未建立足够证据。外部赛果没有进入本包的自动结果。
- 机位分类：当前 3 段罚球者背后、7 段右侧斜机位为候选；不是精确 45 度标定。
- 两类慢动作：均未获得足够对齐证据；197 段速度均 unknown。时间轴覆盖不等于机位识别准确率。
- 脚踝控制/锁踝刚度：现有 COCO17 关键点无法可靠测量。

所有结果仍需要真实复核集。没有新增人工标签，也没有用用户粗看步数调阈值。
'''
    text('06_pending/README.md', pending)
    readme = f'''# Vision II 当前可交付成果包 — 2026-10-10

先打开 `index.html`。这是“工程基本可用 + 有边界的分析候选”汇总，**不是项目全部完成版，也不是识别准确率已验证版**。
原工程：`{ROOT}`。所有原视频、旧结果和用户文件均保留，本包仅复制与新增。

## 1. 可以交付什么

| 类别 | 文件 | 可确认的完成内容 | 不能声称 |
|---|---|---|---|
| 视频与清单 | `01_results/video_index_39.json`；`02_video_review/camera/videos/` | 标准39条、H.264完整可播放副本；原始哈希有记录 | 副本与原始编码字节相同；切片一定包含全部助跑/回放 |
| 步数与步伐呈现 | `01_results/steps_candidates_39.json`；`steps_raw/` | 39条候选可重放，计数分区守恒；连续13、末段调整4、未定22 | 39条步数均正确；候选峰谷等于真实落脚 |
| 门将条件较好的子集 | `01_results/goalkeeper_qa_accepted_15.json`；`goalkeeper_raw/` | 15条通过当前规则和固定1秒观察窗口QA，含踢球前移动、扑救方向/高度候选 | 身份/扑救方向已人工确认；15/39是准确率 |
| 机位时间段与复核工具 | `01_results/camera/`；`02_video_review/camera/index.html` | 39条共197个连续时间区间，CSV/JSON、跳转、逐帧、片段停止可用 | 四类机位都完成，或覆盖率就是准确率 |
| 代码和测试证据 | `03_source/`；`04_validation/` | 源码快照、35项定向测试通过、39条步数缓存重放与文件哈希核对 | 新机器已经一键完成全部原视频推理 |

## 2. 怎么打开

1. **先完整解压 ZIP**，不要在压缩软件内部打开 HTML。
2. 双击 `index.html` 看汇总与39条结果，每条可打开详情并播放视频。JSON可直接下载/查看。
3. 机位页若浏览器限制本地视频，使用本机只读服务器。在包目录执行：

```powershell
python -X utf8 serve_review.py --directory . --port 8885
```

随后打开 `http://127.0.0.1:8885/index.html`；机位复核为 `http://127.0.0.1:8885/02_video_review/camera/index.html`。
也可运行 `start_review.ps1 -PythonExe 'C:\\Users\\30605\\Virtual_Environment\\myenv\\Scripts\\python.exe'`。
服务器只监听本机，无上传接口。关闭终端或 Ctrl+C 停止。不需要下载模型或安装深度学习依赖来浏览成果。

## 3. 文件校验和缓存复现

```powershell
python -X utf8 verify_delivery.py
```

校验包内全部文件 SHA-256、39份计数分区、197段时间轴，以及用同一已记录规则重算39条步伐分类。
该检查仅依赖 Python 标准库（Python 3.11+），可在移动目录后执行；不读取原工程的绝对路径，也不写标签。
这是缓存结果重放，不是从视频重新运行姿态模型。归档中的历史绝对路径保留为来源证据，**不冒充可移植运行配置**。

## 4. 运行依赖与源码范围

`requirements.txt` 是实际运行环境的已观察版本快照，不是新环境安装成功的锁文件。
当前机使用 Python 3.14.7；详细位置见 `04_validation/runtime_observed_2026-10-09.json`。
浏览、文件校验、步伐缓存重放不需要这些重型包；重跑模型才涉及 NumPy/SciPy/OpenCV/Torch/Ultralytics 等。
全项目 unittest 复测为185项通过，1个模块因缺少 pytest 导入失败；普通 pytest 函数不由 unittest 收集。
`requirements-tests-optional.txt` 仅声明可选测试依赖，本包未安装，也未声称全 pytest 测试通过。

`03_source/` 为 src/scripts/tests/assets/configs 的当前源码快照，其中也有历史或实验脚本。**不要无差别批量运行**。
主要代码：`src/style_aware_steps.py`、`src/terminal_footwork.py`、`src/goalkeeper_observation_guard.py`、
`src/camera_timing_v2.py`；入口分别见对应 `scripts/run_*.py`、`scripts/run_camera_timing_analysis.py`。
模型权重、完整比赛视频、所有中间姿态/检测缓存、原大Notebook和虚拟环境未复制，仍在原工程。
39条原编码切片位于原工程 `data/raw_clips/`，本包的视频是同片段的兼容播放副本。
因此全链条重新推理仍应在原工程进行，并使用新输出目录；不是这个精选包的一键功能。

## 5. 解释结果时的必备口径

- “基本没问题”指可打开、可播放、输出结构一致、能复算，不代表人物或动作语义100%正确。
- 连续助跑/末段调整描述某一次罚球，不是给球员永久贴标签。方式未定仍可保留步数候选。
- `contact_guard` 候选保留在总数中，不能直接认作真正最后落脚；null表示未估计，不是0。
- Brozovic 的2步只覆盖已知可见后段，不是完整助跑只有2步；其它片段也没有经人工确认完整覆盖。
- 参数范围只是阈值与触球锚点敏感性，不是置信区间或已证明的允许误差。
- 门将方向为画面左右；原始门将证据中 `auto_accepted=true` / `review_required=false` 只是旧QA标志，不是人工确认。
  本包汇总明确保留 `review_required=true`、`identity_verified=false`。
- 机位分类只有3段背后、7段右斜候选；所有慢动作速度状态仍 unknown。没有人工真值，准确率仍未知。
- 球门正面变换、自动进球判断等未完成项目见 `06_pending/README.md`，没有混进主要成果表。

## 6. 每部分代码的来源

OpenCV提供视频读取、颜色/边缘/几何基础算子；Ultralytics预训练YOLO提供人物/球/姿态候选；
NumPy/SciPy负责数值和信号处理。项目的角色关联、触球搜索、候选计数/步伐分类、观察窗口QA、
机位规则和证据导出是本地项目代码的工程组合。本包不声明这些是 SoccerNet 官方实现或本项目训练的模型。
可用 `03_source/` 逐行说明规则。API/模型、课堂思想与项目自己写的启发式逻辑需要分开陈述。

## 7. 文件来源与测试

`manifest.json` 记录每个交付文件的大小、哈希及复制来源（生成文件不伪造外部来源）。
`04_validation/delivery_build_checks.json` 记录本次新做的代码哈希、原视频及副本核对；
`04_validation/historical_step_validation_2026-10-09.json` 明确是历史验证；复测记录则为2026-10-10。
结果从未被改写为人工真值；历史49行示例CSV、未经来源确认的 manual/confirmed 表、外部赛果表均未列入主要交付。
'''
    text('README.md', readme)
    text('03_source/SOURCE_SCOPE.md', '# 源码快照\n\n此目录保留当前代码以便解释和检查，不是携带全部原始数据/权重/缓存的完整项目镜像。请先阅读包根 README 的第4节。实验/历史脚本不代表已完成模块。\n')
    overview = f'''<p class="muted">IND5003 / VISION II · 2026-10-10</p><h1>当前可交付成果汇总</h1>
<p class="notice"><strong>工程可用，不等于识别准确。</strong> 本包整理基本可用的工具、数据和可复算候选；没有人工真值，不宣称项目全部完成。原工程与原视频均未覆盖。</p>
<div class="cards"><div class="card">视频与索引<h2>39 条</h2>已验证兼容副本，可播放</div><div class="card">步数分析<h2>39 条候选</h2>13 连续 / 4 末段调整 / 22 未定</div><div class="card">门将子集<h2>15 条候选</h2>规则 QA 接受，身份仍待核</div><div class="card">机位时间段<h2>197 段</h2>时间轴完整，分类仍部分未定</div></div>
<h2>先用这些文件</h2><ul><li><a href="README.md">README：方法、限制、启动及依赖</a></li>
<li><a href="01_results/steps_candidates_39.json">39条步数汇总 JSON</a> · <a href="01_results/goalkeeper_qa_accepted_15.json">15条门将候选 JSON</a></li>
<li><a href="02_video_review/camera/index.html">机位时间段 / 逐帧复核页面</a> · <a href="01_results/camera/camera_timings_candidates.csv">机位时间表 CSV</a></li>
<li><a href="04_validation/retest_report.html">复测报告</a> · <a href="04_validation/delivery_build_checks.json">本次整理校验</a> · <a href="requirements.txt">依赖快照</a></li></ul>
<h2>39条逐视频结果</h2><p>点击视频名播放原片段的兼容副本，并查看本条详细证据。步数、方式及门将均为自动候选；不是人工答案。参数范围不是置信区间。</p>
<div class="scroll"><table><thead><tr><th>视频 / 详情</th><th>步数候选</th><th>方式候选</th><th>参数范围</th><th>门将子集</th><th>完整助跑</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<h2>明确不列为已完成</h2><p>可靠左右脚、真实地面俯视轨迹、球真实终点、自动 goal/no-goal、两类慢动作。<a href="06_pending/README.md">查看待完成说明</a>。</p>
<h2>验收口径</h2><p>相关35项单元测试通过；全项目185项通过、1个模块因pytest缺失未载入。39条视频完整解码及本次复制哈希通过。规则QA、重放一致和哈希一致均不是真实视频准确率。</p>
<p>打开复核页建议使用 <code>python serve_review.py --directory . --port 8885</code>，再访问本机8885端口。无需上传、联网或下载模型。</p>'''
    text('index.html', page('Vision II 当前可交付成果汇总', overview))

    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.links=[]
        def handle_starttag(self, tag, attrs):
            self.links += [v for k,v in attrs if k in ('href','src') and v]
    for p in destination.rglob('*.html'):
        if '03_source' in p.parts:
            continue
        parser=Links();parser.feed(p.read_text(encoding='utf-8'))
        for link in parser.links:
            if ':' in link or link.startswith('#'):
                continue
            assert (p.parent / link.split('#')[0].split('?')[0]).exists(), (p,link)

    manifest=[]
    for p in sorted(destination.rglob('*')):
        if p.is_file():
            rel=p.relative_to(destination).as_posix()
            manifest.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p),**provenance.get(rel,{'generated_by':'scripts/build_current_delivery.py'})})
    dump(destination/'manifest.json', {'schema':'visionii-curated-delivery-v1','files':manifest,'manifest_excludes_itself':True})
    checked=subprocess.run([sys.executable,'-X','utf8',str(destination/'verify_delivery.py')],capture_output=True,text=True,encoding='utf-8')
    print(checked.stdout,flush=True)
    if checked.returncode:
        raise RuntimeError(checked.stderr + checked.stdout)
    archive=destination.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'x',allowZip64=True) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                compression=zipfile.ZIP_STORED if p.suffix.lower() in ('.mp4','.jpg','.png','.zip') else zipfile.ZIP_DEFLATED
                z.write(p,destination.name+'/'+p.relative_to(destination).as_posix(),compress_type=compression)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(manifest)+1
    record={'directory':str(destination),'archive':str(archive),'archive_bytes':archive.stat().st_size,
            'archive_sha256':sha(archive),'zip_crc_pass':True,'offline_validation':json.loads(checked.stdout),
            'original_source_videos_rechecked':39,'copied_review_videos':39,'human_ground_truth_created':False}
    dump(destination.parent/(destination.name+'_package_check.json'),record)
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'deliverables/VisionII_Current_Deliverable_2026-10-10')
    build(p.parse_args().output.resolve())
