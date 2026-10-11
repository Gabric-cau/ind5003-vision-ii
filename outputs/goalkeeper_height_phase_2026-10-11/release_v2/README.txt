守门员完整首次扑救阶段与高度探索包。先打开report.html。
正式方向/人工方向/高度/整条QA未被覆盖；summary.json中的phase_height_candidate和silhouette_height_proxy_candidate为不同证据等级的新增字段。
严格高度：可见肩髋姿态、稳定预踢地面参考、落地前伸展峰值，缺口/深度偏移拒绝。
较弱轮廓高度：连续SAM2人物轮廓、变扁/下边缘趋稳落地候选、接地锚点之间直线投影地面路径假设；不是真实髋点、精确离地高度或地面真值。
高度低/中/高按预踢身体像素高度归一化，非米制；未做视频人工高度真值标注，accuracy_estimate=null。
相机运动为排除人物的LK+RANSAC背景二维仿射；不能解决所有景深运动和遮挡。
跟踪至踢球后2秒或实际视频结尾；遇掩膜QC、镜头突变/ROI出界停止身份连接；不自动桥接到另一个人物。首次扑救完整候选不等于整个长视频都跟完。
代码复现（在原项目环境，权重和上游缓存必须存在）：
python -X utf8 scripts/run_goalkeeper_height_phase.py --output-dir outputs/new_phase_run
默认只处理23条高度unknown；使用--clips传入规范39条clip_id可复现全批次。
--cache-run仅在视频JPEG字节、窗口、ROI、初始提示和模型哈希一致时复用本地实验缓存；正式结果不作为高度标签输入。
SAM2：Meta官方sam2.1_hiera_tiny.pt，512分辨率CPU实验（非默认1024等效精度声明）。
本地依赖隔离：.deps_goalkeeper_segmentation；姿态Ultralytics由原项目.venv导入。
包含局部JPEG、掩膜、候选姿态、阶段视频、源码快照与哈希。完整原视频、150MB权重和上游全部缓存未打包。
模型来源：https://github.com/facebookresearch/sam2
阶段分类/阈值/角色筛选/质量门控为本项目工程实现，不声称来自官方SAM2或SoccerNet。
