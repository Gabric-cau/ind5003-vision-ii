"""Package experiment evidence, not a new default goalkeeper classifier."""
from pathlib import Path
import sys, io, unittest, json, html, shutil, zipfile, importlib.metadata
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_mask_comparison import read,save,sha
OUT=ROOT/'outputs/goalkeeper_mask_comparison_2026-10-11'


def main():
    first=read(OUT/'experiment_v1/classical_results.json')
    repeat=read(OUT/'repeatability_v1/classical_results.json')
    improved=read(OUT/'chroma_gated_v2/classical_results.json')
    sam=read(OUT/'experiment_v1/sam2_results.json')
    visual=read(OUT/'ai_visual_review.json')
    source=read(OUT/'experiment_v1/protocol.json')
    suites=['tests.test_goalkeeper_mask_experiment','tests.test_goalkeeper_role_association',
            'tests.test_goalkeeper_pose_direction','tests.test_goalkeeper_direction_evidence',
            'tests.test_goalkeeper_observation_guard']
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    (OUT/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    mask_checks={}
    for method in ['color','grabcut']:
        paths=list((OUT/'experiment_v1/masks'/method).glob('*.png'))
        mask_checks[method]=all(sha(p)==sha(OUT/'repeatability_v1/masks'/method/p.name) for p in paths)
    unchanged={p:sha(Path(p))==v for p,v in source['input_hashes'].items()}
    # Preserve old v1 snapshot before the ablation edits; freeze v2 current source.
    files=['src/goalkeeper_mask_experiment.py','scripts/run_goalkeeper_mask_comparison.py',
           'scripts/finalize_goalkeeper_mask_comparison.py','tests/test_goalkeeper_mask_experiment.py']
    for name in files:
        dest=OUT/'source_snapshot_v2'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    summary=dict(clip_id=source['clip_id'],frame_window_inclusive=source['frame_window_inclusive'],
       experiment_scope='one clip pilot, not 39-clip accuracy evaluation',
       methods=['Lab color likelihood','GrabCut','camera-compensated LK optical flow','SAM2.1 Hiera tiny video mask propagation'],
       default_results_not_modified=all(unchanged.values()),baseline_direction_still='unknown',
       existing_unknown_count_not_changed=6,no_human_ground_truth=True,accuracy_estimate=None,
       tests_run=result.testsRun,tests_pass=result.wasSuccessful(),classical_repeat_equal=first['methods']==repeat['methods'],
       repeat_mask_sha_equal=mask_checks,sam2_elapsed_seconds=sam['elapsed_seconds'],
       sam2_completed_frames=len(sam['frames']),input_hash_checks=unchanged,
       model_sha256=sam['checkpoint_sha256'],ai_visual_review=visual,
       sam2_source=dict(url='https://github.com/facebookresearch/sam2',commit='2b90b9f5ceec907a1c18123530e92e794ad901a4'),
       implementation_notes=['Custom automatic role initialization, color score and QC are project heuristics, not claimed to come from SoccerNet.',
          'Classical methods reanchor through 111; SAM2 receives only frame 87 bbox. This is a pilot comparison, not an equal-prompt benchmark.',
          'SAM2 crop JPEG inputs and masks retained; no model weights or installed packages bundled.',
          'Video and unit-test QA do not establish goalkeeper identity or dive-direction accuracy.'])
    save(OUT/'summary.json',summary)
    rows=''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{html.escape(r["finding"])}</td><td>{html.escape(r["limitation"])}</td></tr>' for r in visual['method_findings'])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>门将分割方法对比实验</title>
<style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:36px auto;padding:0 20px;background:#fafafa;color:#17232b}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border:1px solid #ccd3d9;text-align:left}}video,img{{width:100%;background:#111}}.warn{{padding:14px;background:#fff0d3}}code{{word-break:break-all}}</style>
<h1>门将与草坪：四种方法实际对比</h1><p>Cuadrado 点球；原视频 2.90–4.60 秒，52 帧，门将原姿态检测最后一帧为 111（3.70 秒）。裁判红框来自原检测，只作身份/遮挡提示，不是人工真值。</p>
<p class="warn">本报告只证明这段视频里方法是否得到候选分割，不证明 39 条准确率。未覆盖默认结果；6 条 unknown 仍保留；没有人工标签或自动填补被遮挡身体。</p>
<table><tr><th>方法</th><th>本次观察</th><th>限制</th></tr>{rows}</table>
<h2>第一轮：四方法并排</h2><p>左上 Lab 颜色；右上 GrabCut；左下 LK 光流；右下 SAM 2。黄色是候选掩膜/特征点，不代表方向已验证。</p>
<video controls src="experiment_v1/cuadrado_mask_comparison.mp4"></video>
<h2>第二轮：排除白色门线干扰</h2><p>颜色模型加入 HSV 饱和度 ≥60 的工程过滤，光流特征也检查球衣颜色。该阈值为本段试验设置，未证明适合所有球衣；右下未再次运行 SAM 2（其结果见上方）。</p>
<video controls src="chroma_gated_v2/cuadrado_mask_comparison.mp4"></video>
<h2>关键帧与复核边界</h2><img src="experiment_v1/comparison_f120.jpg"><p>AI 静态查看帧：{visual['frames_inspected']}；这是 AI 视觉检查，不是用户人工复核或逐像素标注。</p>
<h2>流程和代码来源</h2><ol><li>读取已有门线角色筛选结果，用站立门将自动初始化。未读取人工身份/进球标签。</li>
<li>根据预踢检测框生成扩大局部 ROI；87–95 帧肩/髋内侧生成球衣样本，周围非人物绿色区域生成草坪样本。</li>
<li>颜色：项目编写的 Lab 对角高斯似然与连通区域/连续性筛选；第二轮追加低饱和度排除。它不是通用语义人体分割。</li>
<li>GrabCut：直接调用 OpenCV；可靠姿态产生前景种子，漏检后用 Farneback 后向光流传播前一掩膜，并用颜色约束更新。不能称为完全独立于颜色/光流的消融。</li>
<li>LK：OpenCV 金字塔光流、前后向误差&lt;1.5px；非人物背景 RANSAC 估计镜头仿射运动。跟踪点可能漂移，第二轮增加球衣颜色检查后主动拒绝失效点。</li>
<li>SAM 2：Meta 官方视频预测器、SAM2.1 Hiera tiny、CPU；只使用第87帧自动门将框，后面不加人工提示。对局部裁剪逐帧传播掩膜。</li>
<li>所有结果单独保存并核对原视频/姿态/默认汇总 SHA256。四方法提示数量不同，本次不是严格的同提示数量基准测试。</li></ol>
<p>出处：<a href="https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html">OpenCV GrabCut</a>；<a href="https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html">OpenCV 光流</a>；<a href="https://github.com/facebookresearch/sam2">Meta SAM 2</a>。本次没有引入 SoccerNet 算法代码。</p>
<h2>验证与交付</h2><p>{result.testsRun} 项所选单元测试通过：{result.wasSuccessful()}；经典方法重跑结构一致：{summary['classical_repeat_equal']}；掩膜文件哈希一致：{all(mask_checks.values())}。SAM 2 完成{len(sam['frames'])}帧，CPU 用时{sam['elapsed_seconds']:.1f}秒。每个对比视频52帧全部解码通过。以上不是识别准确率。</p>
<p><a href="summary.json">实验汇总</a> · <a href="ai_visual_review.json">AI视觉复核记录</a> · <a href="README.txt">复现说明</a> · <a href="requirements-mask-experiment.txt">依赖说明</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    readme='''门将掩膜/光流对比：实验包，不替换正式结果。
项目根目录：C:\\Users\\30605\\PycharmProjects\\JupyterProject
先打开 report.html。experiment_v1 包含初版经典方法和 SAM2；chroma_gated_v2 包含颜色/光流改进试验；repeatability_v1 为经典方法复现。
分割得到的是可见区域，不是被遮挡完整身体；没有人工 ground truth，不能报告 accuracy。
原视频仍在 data/raw_clips/col_eng_p03_cuadrado_col11.mp4。包中只含局部复核视频，不含完整原视频、模型或安装依赖。
复现需在原项目根目录，保留原姿态缓存和角色实验结果。选择新的输出目录防止覆盖：
python -X utf8 scripts/run_goalkeeper_mask_comparison.py --output-dir outputs/my_mask_run
python -X utf8 scripts/run_goalkeeper_mask_comparison.py --output-dir outputs/my_mask_run --mode sam2 --checkpoint models/weights/sam2.1_hiera_tiny.pt
python -X utf8 scripts/run_goalkeeper_mask_comparison.py --output-dir outputs/my_chroma_run --chroma-gate
python -X utf8 -m unittest tests.test_goalkeeper_mask_experiment
官方权重：https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
SAM2 官方源码提交：2b90b9f5ceec907a1c18123530e92e794ad901a4
隔离目录 .deps_goalkeeper_segmentation；未更新原 torch/opencv 环境。
本机 SAM2 CUDA 扩展未编译，apply_postprocessing=False；CPU 运行。ANTLR 4.9.3 实现修复了最新ANTLR的不兼容；不要仅以残留dist-info识别版本。
source_snapshot_v2 为改进轮代码；experiment_v1/source_snapshot 为第一轮经典方法源码。SAM2使用第一轮进程已加载的源码。
'''
    (OUT/'README.txt').write_text(readme,encoding='utf-8')
    (OUT/'requirements-mask-experiment.txt').write_text('''# Existing validated CPU runtime; use a separate environment, do not blindly upgrade the project.
numpy==2.4.6
opencv-python==4.13.0.92
torch==2.13.0
torchvision==0.28.0
# SAM2 helper dependencies installed in isolated target; ANTLR pin is important.
hydra-core==1.3.7
omegaconf==2.4.0
antlr4-python3-runtime==4.9.3
iopath==0.1.10
portalocker==4.4.0
tqdm
Pillow
PyYAML
# Install official SAM2 separately with SAM2_BUILD_CUDA=0 and --no-deps --no-build-isolation:
# git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
# FFmpeg binary is required for H.264 review exports, not a Python pip dependency.
''',encoding='utf-8')
    if not result.wasSuccessful() or not all(unchanged.values()) or not all(mask_checks.values()):
        raise ValueError('Experiment validation failed')
    bundle=ROOT/'deliverables/VisionII_Goalkeeper_Mask_Experiments_2026-10-11.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.name!='comparison_staging.mp4':z.write(p,p.relative_to(OUT))
    save(OUT/'bundle_validation.json',dict(path=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size))
    print(json.dumps(dict(tests=result.testsRun,baseline_unchanged=all(unchanged.values()),bundle=str(bundle)),ensure_ascii=False))


if __name__=='__main__':main()
