Vision II — 2026-10-11 增量优化实验包
============================================================

先打开 report.html。不要用这个实验包替换昨天的成果包。

结果性质
--------
本包没有人工真值。visual_qa/ai_findings.json 是 Codex 的静态画面诊断，
不是用户复核、不是独立标注，也不能计算真实准确率。
54段初始空间候选 -> 38段自动防护后候选 -> AI明确隔离2段已知疑点。
剩余36段依然只是待核查候选，所有播放速度保持 unknown。
自动防护减少了观众/教练/同队多人干扰，也误拒了一些真实球员特写。
本轮没有证明慢动作识别成功率提升。

目录
----
measurements/             39条 canonical clips 的本轮姿态观测缓存；3593帧
evaluation_v1/            完整/部分姿态对照、有序时间缩放、负对照、原始结果
dense_views_v1/           镜头内部稠密空间证据，尚未身份/场景排除的54段
guarded_views_v2/         自动场景防护 + 守门员分支回退修复后的38段
goal_tracking_pilot_v1/   Cuadrado单镜头光流实验；不能外推为39条全部改进
visual_qa/               54段首/中/末采样拼图与明确署名AI的诊断记录
review_candidates.json   36段待核查 + 2段AI隔离；自动结果和AI隔离分开
validation/              单元测试日志、源视频和旧成果ZIP哈希检查
comparison_summary.json  对照统计；不是准确率
examples/                两段恢复时间区间的H.264短视频与解码检查

代码位置（从项目根目录看）
------------------------
src/camera_partial_alignment.py       部分关节与有序时间缩放匹配
src/camera_view_intervals.py          保守连续区间
src/camera_view_guard.py              颜色/场景保护及分支修复（实验模块）
src/goal_temporal_tracking.py         短时球门图像几何跟踪（实验模块）
scripts/run_camera_partial_pose_experiment.py
scripts/evaluate_camera_partial_alignment.py
scripts/run_camera_dense_view_experiment.py
scripts/run_goal_temporal_pilot.py
scripts/run_camera_guarded_views.py
scripts/render_camera_interval_qa.py
scripts/finalize_camera_optimization.py
scripts/package_camera_optimization.py
tests/test_camera_partial_alignment.py
tests/test_camera_view_intervals.py
tests/test_camera_view_guard.py
tests/test_goal_temporal_tracking.py

环境与复现
----------
这是增量包，不是完整独立项目：需原项目的 src/scripts、既有依赖环境、
models/weights/yolo11n_pose.pt、原始视频及 outputs/camera_timing_retest_2026-10-10_01。
ZIP不包含视频全集、模型权重、全部历史上游输出或虚拟环境。
本轮未安装新库，主要用已有 numpy、opencv-python、torch、ultralytics。
环境依赖仍以原项目 requirements.txt 及已有环境为准；包中环境版本快照用于追踪。

工作目录：C:\Users\30605\PycharmProjects\JupyterProject
本轮解释器：C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe

实际运行顺序（在干净的输出位置运行，已存在的实验目录禁止覆盖）：
python -X utf8 scripts/run_camera_partial_pose_experiment.py --cohort all
python -X utf8 scripts/evaluate_camera_partial_alignment.py
python -X utf8 scripts/run_camera_dense_view_experiment.py
python -X utf8 scripts/render_camera_interval_qa.py
python -X utf8 scripts/run_goal_temporal_pilot.py
python -X utf8 scripts/run_camera_guarded_views.py
python -X utf8 scripts/finalize_camera_optimization.py
python -X utf8 scripts/package_camera_optimization.py

有现成缓存时，第一步可用 --resume（先核对 extraction_protocol.json）。
评价脚本可用 --name 指定新的评价目录；稠密视角脚本可用 --output-dir。
试点、guard及打包脚本默认拒绝覆盖。想重跑它们，应先复制脚本并设置新的输出根目录，
不要删除/覆盖本包缓存，也不要将不同协议的缓存混在同一目录。
每个实验 protocol.json 保存运行时源代码/输入哈希；压缩包 MANIFEST.json 保存文件哈希。

局限和优先级
------------
1. 球衣颜色不等于球员身份：需要号码证据或可信跨帧关联，尤其同队队员/观众。
2. 二值草地掩膜只能作场景提示，不能排除所有穿球衣观众，亦会误拒无草地的球员近景。
3. 球门角度对角点与未知内参敏感；右斜方候选不能叫精确45度标定。
4. 缺失腿部姿态和跨视角变形仍阻碍慢动作配对；不同事件动作相似性也可能误导。
5. 本轮只分析预选动作窗口，不是所有视频每一帧都已完成空间/身份验证。
6. 球门光流仅跟踪图像平面，不推导球进没进、真实3D球速或不存在的新机位。

来源
----
通用时间缩放/子序列匹配思路：
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_DiagonalMatching.html
Shi-Tomasi / Lucas-Kanade OpenCV接口：
https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
项目自行实现掩码、阈值、防护、组合流程及对照；这些来源没有验证本项目足球结果。
