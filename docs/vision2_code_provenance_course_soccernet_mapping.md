# Vision II 代码来源、课程对应与 SoccerNet 对照说明

> 2026-10-08 更正：旧文中的“人工确认”没有真人复核依据。步数、触球帧和守门员种子标签来自此前AI助手任务，现已降级为未验证AI候选；不能作为独立真值。旧版准确率/MAE/RMSE表述撤回。下文源码行号为原审计快照，新增优化后可能移动。更新记录见 `vision2_optimization_audit_2026-10-08.md`。

## 1. 结论先行

本项目与课程教材、OpenCV Bootcamp 和 SoccerNet 存在大量**概念与任务层面的重合**，但三者不能混为同一种“代码来源”。正确的来源说明应分成四层：

1. **课程与教材提供方法框架**：图像/视频读取、颜色空间、掩膜、几何变换、目标检测、目标跟踪、姿态估计等。
2. **开源库提供底层实现**：OpenCV、Ultralytics YOLO、NumPy、SciPy、Pandas 和 PyYAML。
3. **SoccerNet提供足球视频任务的研究参照**：镜头边界检测、多人跟踪、球场标定、Ball Action Spotting 和 Game State Reconstruction。
4. **本项目自行定义点球专用逻辑**：罚球者选择、触球帧候选、左右脚判断、助跑起点、步数信号、守门员筛选与分类、足球候选评分、运动约束 beam search、人工复核和 QA 状态。

经当前工作目录检查：

- `src/`、`scripts/` 和 `requirements.txt` 中没有导入 `SoccerNet`、`TrackLab`、StrongSORT 或 SoccerNet DevKit。
- `requirements.txt` 中没有 SoccerNet 相关依赖。
- 项目跟踪器明确写明不依赖 ByteTrack/LAP，而是本项目的确定性最近中心/尺度匹配。
- 对 `_NearestCentreBoxTracker`、`goal_line_seed_plus_temporal_stitching`、`contact_anchor_fallback`、`support_endpoint` 等项目专用标识进行公开网页检索，没有找到外部同名实现。
- 因此，目前可以说“**没有发现直接复制 SoccerNet 代码的证据**”；但由于该文件夹不是 Git 仓库，无法通过 commit 历史完成严格的逐行来源取证，所以不应使用“已经证明绝无任何相似代码”这种绝对表述。

建议在答辩中使用下面这句话：

> The project is course-inspired and library-assisted, but the penalty-specific decision logic is implemented locally. SoccerNet is used as a conceptual benchmark for task decomposition, not as an imported codebase.

中文表达：

> 本项目的方法框架来自课程中的计算机视觉知识，底层能力由OpenCV、Ultralytics和SciPy等开源库提供；点球场景中的特征定义、规则组合和质量控制由本项目实现。SoccerNet用于任务设计和方法对照，没有作为直接导入的代码库。

---

## 2. 核对范围与证据边界

本次核对使用了以下材料：

- 教材：Vik Gopal 与 Lim Tiong Wee，*Data Analytics for Sense-Making*，第9章 Computer Vision，PDF第201-216页。
- 教材第216页链接的官方 OpenCV Bootcamp 公开课程大纲。
- 本项目当前 `src/`、`scripts/`、`tests/`、`requirements.txt`、README 和主 notebook。
- SoccerNet 官方网站的任务定义，以及官方 `sn-tracking`、`sn-calibration`、`sn-gamestate` 仓库。
- Ultralytics、OpenCV 和 SciPy 官方文档。

没有收到老师实际发放的 Bootcamp notebook、Pose Estimation notebook 或 Trajectory Modelling notebook，因此课程代码只能依据教材和公开课程大纲做**概念级对应**，不能声称完成了与老师私人课程文件的逐行比较。如果之后取得这些 notebook，应再运行一次逐文件 diff/相似性检查。

---

## 3. 与教材第9章的重合部分

### 3.1 教材内容概览

教材第9章包含以下内容：

- PDF第201页：CV应用、预训练深度学习模型与预处理的重要性。
- PDF第202页：使用 `cv2.imread` 读取图像，以及OpenCV的BGR通道顺序。
- PDF第203-204页：灰度化、平移、旋转、resize和 `cv2.warpAffine`。
- PDF第204-207页：HSV颜色选择、mask、`cv2.inRange`、轮廓和凸包，并以足球为例。
- PDF第207-209页：`cv2.getPerspectiveTransform`、`cv2.warpPerspective`，其中直接包含足球转播画面的俯视变换示例。
- PDF第210-212页：Sobel、Canny和Hough圆检测，再次以足球为例说明仅依靠圆形会产生误检，需要上下文排除错误候选。
- PDF第213-215页：目标检测、目标分类、目标跟踪、姿态估计和预训练模型调用。
- PDF第216页：OpenCV Bootcamp、OpenCV文档、Szeliski教材和OpenCV仓库。

### 3.2 教材与本项目的具体对应

| 教材知识 | 本项目位置 | 重合程度 | 本项目在教材基础上的扩展 |
|---|---|---:|---|
| 图像/视频读取 | `src/video_utils.py:5-61`、各 pipeline 的 `cv2.VideoCapture` | 直接对应 | 增加逐帧时间戳、manifest切片和统一 clip ID |
| Resize与ROI | `src/step_pipeline.py:319-396, 444-535` | 直接对应 | 根据全帧检测质量触发960px局部高分辨率回退，并映射回原图坐标 |
| BGR/HSV与颜色掩膜 | `src/camera_angles.py:30-45`、`src/ball_pipeline.py:137-155`、`src/goalkeeper_pipeline.py:265-296` | 直接对应 | 不直接用颜色决定类别，而是作为球/草地区域的弱证据与评分项 |
| 仿射变换 | 教材 `warpAffine`；项目 `src/goalkeeper_pipeline.py:55-118` | 方法相关 | 项目从背景光流估计逐帧仿射运动，用逆变换稳定守门员轨迹 |
| 透视变换/单应性 | `src/homography.py:6-35` | 直接对应 | 封装 `findHomography(RANSAC)` 与 `perspectiveTransform`；当前核心39次点球结果并未完成统一球场标定 |
| Canny + Hough直线 | `src/camera_angles.py:37-43` | 直接对应 | 用球场直线主方向构造早期机位分类启发式；不是当前核心特征提取入口 |
| Hough圆与足球上下文 | 教材第211-212页；项目没有直接调用 `HoughCircles` | 概念对应 | 项目改为YOLO sports-ball候选，再结合白色、绿色背景、形状和运动连续性排除误检 |
| 目标检测 | `src/ball_pipeline.py:158-193, 386-512` | 直接对应 | 使用COCO预训练YOLO11检测 class 32 sports ball，并加入点球场景约束 |
| 目标跟踪 | `src/step_pipeline.py:568-652`、`src/goalkeeper_pipeline.py:297-396` | 任务对应 | 自行实现轻量、确定性的数据关联与跨ID拼接，不调用教材模板tracker |
| 姿态估计 | `src/step_pipeline.py`、`src/kicker_analysis.py`、守门员fallback | 直接对应 | 从17个COCO关键点中使用髋部与脚踝构造点球动作特征 |

最重要的教材联系是第212页提出的“可以用什么上下文排除错误足球”。本项目的答案是：

- 足球应在触球时靠近可信脚踝锚点；
- 触球后应具有明显速度和连续方向；
- 轨迹总体应朝守门员/球门区域前进；
- 外观上结合球框白色比例、周围草地绿色比例和框形圆度；
- 低覆盖、低位移、方向不合理或缺少球门锚点的样本进入人工复核。

这部分是对教材问题的项目化解决，不是照搬教材代码。

---

## 4. 与 OpenCV Bootcamp、Pose Estimation 和 Trajectory Modelling 的对应

### 4.1 OpenCV Bootcamp

官方公开大纲包含14个模块，其中与项目直接相关的是：

- Images与basic image manipulation；
- Read/write video；
- Image filtering and edge detection；
- Image features and alignment；
- Object tracking；
- Object detection；
- Pose estimation using OpenPose。

官方课程说明：<https://opencv.org/opencv-free-course/>

课程与项目的关系不是“复制同一个notebook”，而是“课程中的通用模块被组合成点球研究流程”。例如：

```text
Bootcamp video I/O
  -> 逐帧读取39段点球视频

Bootcamp object detection
  -> YOLO sports-ball候选

Bootcamp object tracking
  -> 罚球者、守门员和足球的跨帧关联

Bootcamp image alignment
  -> 背景特征点 + 光流 + 仿射模型的镜头运动校正

Bootcamp pose estimation
  -> 髋部/脚踝关键点序列
  -> 左右脚、触球帧、助跑起点与步数
```

### 4.2 Pose Estimation：相同任务，不同模型

Bootcamp公开大纲使用OpenPose作为教学示例；本项目实际使用Ultralytics YOLO11 Pose。两者都输出人体关节点，但不是同一套模型代码。

本项目使用COCO-17索引：

- 11：left hip
- 12：right hip
- 15：left ankle
- 16：right ankle

Ultralytics官方Pose文档说明了17个关键点、`result.keypoints.xy`、关键点置信度和人物检测框：<https://docs.ultralytics.com/tasks/pose/>

代码来源应写成：

> Human keypoints were obtained using a COCO-pretrained Ultralytics YOLO11 pose model. The subsequent temporal feature engineering and penalty-specific classification rules were implemented for this project.

不能写成“我们训练了姿态模型”，因为当前权重是预训练模型；也不能写成“OpenPose输出”，因为实际并未调用OpenPose。

### 4.3 Trajectory Modelling：本项目到底做了什么

本项目包含三类轨迹：

1. 罚球者髋部和脚踝的2D像素轨迹；
2. 守门员中心点的、按身体高度归一化并进行镜头运动校正的轨迹；
3. 足球检测框中心的2D像素轨迹。

足球部分使用的是**一阶运动连续性模型 + beam search数据关联**：

- 根据前一速度预测下一位置；
- 对预测残差过大的候选进行门控；
- 用指数式速度更新平滑瞬时速度；
- 在每帧保留有限数量的高分路径；
- 最后加入朝向球门的位移和方向一致性评分。

因此它可以称为：

> image-plane trajectory association with a first-order motion prior

但不应称为：

- 物理弹道模型；
- 3D轨迹重建；
- Kalman filter；
- 以m/s或km/h计量的真实球速；
- 经相机标定后的场地轨迹。

当前输出速度是 `px/s`。如果课程中的Trajectory Modelling要求抛物线回归、Kalman filter或真实世界坐标，本项目只是部分重合，还没有完整覆盖该要求。

---

## 5. 与 SoccerNet 的对应关系

### 5.1 强概念对应，但没有直接代码依赖

SoccerNet官方任务覆盖镜头分割、Ball Action Spotting、球员跟踪、球场定位、相机标定和Game State Reconstruction。任务总览见：<https://www.soccer-net.org/tasks>

本项目与其对应如下：

| 本项目模块 | SoccerNet最接近任务 | 相同点 | 关键区别 |
|---|---|---|---|
| 第一镜头截止检测 | Camera Shot Segmentation | 都寻找镜头边界 | SoccerNet是13类逐帧分类和边界检测；本项目只用HSV直方图卡方差找第一个cut |
| 触球帧 | Ball Action Spotting | 都定位球相关动作时间 | SoccerNet从长视频学习12类动作时间戳；本项目只分析已切点球片段，并用脚踝运动候选 + 人工确认 |
| 人物跟踪 | SoccerNet Tracking | 都输出frame、box和track identity | SoccerNet使用完整MOT基准及HOTA；本项目使用短时间窗的贪心中心/尺度匹配，没有Re-ID和HOTA |
| 守门员选择 | Tracking / Game State Reconstruction | 都区分人物轨迹和角色 | SoccerNet识别所有人物、角色、队伍和球衣号码；本项目只在点球场景用位置、草地和动作先验筛选守门员 |
| 单应性模块 | Calibration / Field Localization | 都利用足球场近似平面和homography | SoccerNet检测语义球场线并估计完整相机参数；项目现有homography是轻量封装，未用于当前核心结果的统一场地标定 |
| 足球轨迹 | Tracking / GSR | 都希望跨帧定位球 | SoccerNet的目标是标准MOT/GSR；本项目从触球脚出发，用点球专用外观和运动约束选择轨迹 |
| QA与review状态 | SoccerNet benchmark philosophy | 都强调检测与关联质量 | SoccerNet用HOTA/DetA/AssA等标准指标；项目采用coverage、confidence、warnings和人工复核 |

SoccerNet官方说明：

- Camera Shot Segmentation：<https://www.soccer-net.org/tasks/camera-shot-segmentation>
- Ball Action Spotting：<https://www.soccer-net.org/tasks/ball-action-spotting>
- Tracking：<https://www.soccer-net.org/tasks/tracking>
- Game State Reconstruction：<https://www.soccer-net.org/tasks/game-state-reconstruction>
- Tracking代码：<https://github.com/SoccerNet/sn-tracking>
- Calibration代码：<https://github.com/SoccerNet/sn-calibration>
- Game State代码：<https://github.com/SoccerNet/sn-gamestate>

### 5.2 最接近的总体架构：SoccerNet Game State Reconstruction

SoccerNet GSR官方baseline把流程拆成：

```text
bbox detector
-> Re-ID
-> tracker
-> pitch localization
-> camera calibration
-> jersey number
-> role/team aggregation
```

公开配置使用Ultralytics YOLO检测、Re-ID、StrongSORT类tracker和相机标定模块。我们的结构则是：

```text
manifest clips
-> YOLO pose / YOLO ball detection
-> short-window deterministic tracking
-> penalty-specific identity selection
-> contact-frame anchoring
-> kicker / goalkeeper / ball features
-> manual review and QA
```

两者最明显的共同思想是“先检测，再跨帧关联，再做场景级语义”。区别是研究目标、规模和实现完全不同。

### 5.3 SoccerNet中没有直接对应的项目特征

当前核对的SoccerNet官方任务和baseline中，没有发现以下点球专用实现：

- 通过左右脚踝相对髋部的运动判定射门脚；
- 将左右脚踝的相位投影到助跑方向以计算步数；
- 触球前1.5秒守门员活动程度分类；
- 使用身体高度作为守门员位移尺度；
- 对最后射门脚摆动进行计步抑制，并保留支撑脚endpoint；
- 从触球脚锚点启动足球beam search；
- 将官方结果标签与球轨迹几何分开保存。

这些是本项目的研究设计，不应标成“来自SoccerNet”。

---

## 6. 每部分代码的来源与责任边界

### 6.1 文件级来源表

| 文件/模块 | 外部基础 | 本项目实现 | 使用状态 |
|---|---|---|---|
| `src/video_utils.py` | OpenCV `VideoCapture`、`VideoWriter` | 上下文管理器、元数据、逐帧generator和按秒切片 | 活跃基础模块 |
| `src/auto_slicer.py` | PyYAML | 按manifest循环生成canonical clips及命名规则 | 活跃数据准备 |
| `src/camera_angles.py` | 教材/OpenCV的HSV、Canny、Hough、直方图 | 镜头cut阈值与早期机位分类启发式 | `ShotDetector`活跃；机位分类不是当前核心结果 |
| `src/homography.py` | OpenCV `findHomography`、RANSAC、`perspectiveTransform`；教材第207-209页 | 简单transformer封装 | 可用但当前核心39条结果未完成真实场地标定 |
| `src/step_pipeline.py` | Ultralytics预测接口、OpenCV、COCO-17格式 | ROI回退、候选去重、自定义tracker、罚球者排序、触球帧候选、助跑起点、QA | 活跃核心模块 |
| `src/kicker_analysis.py` | NumPy；SciPy `find_peaks` | 左右脚时序比较、身体尺度归一化、相位信号、步数边界规则、置信度 | 活跃核心模块 |
| `src/goalkeeper_pipeline.py` | OpenCV Shi-Tomasi角点、Lucas-Kanade光流、`estimateAffinePartial2D(RANSAC)` | 守门员候选评分、草地先验、跨track-ID拼接、低质量警告 | 活跃核心模块 |
| `src/goalkeeper_analysis.py` | NumPy统计与向量运算 | pre-kick活动量、扑救方向/高度阈值、coverage/confidence | 活跃核心模块 |
| `src/ball_pipeline.py` | Ultralytics YOLO object detection、OpenCV HSV、NumPy | 脚踝锚点、外观评分、运动门控、beam search、ballistic envelope、QA | 活跃核心模块 |
| `src/ball_outcome.py` | NumPy距离/分位数 | 轨迹摘要、速度/覆盖率、review规则、observed outcome与geometry分离 | 活跃核心模块 |
| `scripts/run_*.py` | Python `argparse` | 固定入口、默认路径、单次加载模型、输出摘要 | 活跃运行入口 |
| `scripts/render_*_qa.py` | OpenCV绘图 | 生成候选/轨迹复核图 | 活跃QA入口 |
| `scripts/merge_manual_step_reviews.py` | CSV/YAML标准库 | 严格检查39个clip ID、人工步数、landing frames和自动结果合并 | 活跃人工复核 |
| `src/eda_analysis.py` | Pandas、Matplotlib、Seaborn | 描述性统计与绘图封装 | 分析辅助 |
| `tests/` | `unittest`/`pytest` | 对自定义规则的合成单元测试 | 逻辑测试，不等于真实视频准确率 |
| `notebooks/vision2_penalty_analysis.ipynb` | 上述库和本项目模块 | 探索过程、人工检查、最终结果展示 | 前150个cell为探索历史；最终可重复流程从“Step Counting (Final Reproducible Pipeline)”开始 |

### 6.2 需要特别说明的早期脚手架

`src/pipeline.py` 不是当前39次点球结果的正式生成器。该文件包含：

- 人工构造的 `sample_traj`；
- 人工构造的守门员和足球轨迹；
- `generate_synthetic_benchmark()` 随机生成39条示例数据；
- 固定的示例homography点。

正式结果使用README列出的脚本：

```powershell
python scripts/run_step_counting.py
python scripts/merge_manual_step_reviews.py
python scripts/rerun_goalkeeper_pose.py --all
python scripts/run_goalkeeper_analysis.py
python scripts/run_ball_outcome_analysis.py
```

答辩时若老师打开 `src/pipeline.py`，应直接说明它是早期项目结构和接口测试用的legacy scaffold，不是实验结果来源。报告不能引用其中的合成轨迹作为真实数据。

---

## 7. 完整技术解法与公式

### 7.1 数据清单与视频切片

`configs/penalty_slicing_manifest.yaml`定义match、penalty number、kicker、team、start/end time。`auto_slicer.py`把时间转换为frame：

\[
f_{start}=\lfloor t_{start}\cdot FPS\rfloor,\quad
f_{end}=\lfloor t_{end}\cdot FPS\rfloor
\]

每个点球用唯一 `clip_id` 连接人工标注、姿态诊断、守门员特征、球轨迹和结果。

来源：OpenCV视频I/O是库能力；manifest结构和canonical ID策略是项目设计。

### 7.2 第一镜头边界

每帧缩小为160×90，转换为HSV，计算H-S二维直方图并归一化，再计算相邻帧直方图的卡方差：

\[
D_t=\chi^2(H_{t-1},H_t)
\]

若 \(D_t\) 超过阈值，则认为发生镜头切换。只分析第一个镜头，并设置8秒上限，避免慢动作回放污染动作特征。

来源：直方图、HSV和 `compareHist` 来自OpenCV；只取第一镜头和8秒上限是项目规则。它与SoccerNet Camera Shot Boundary Detection任务对应，但不是SoccerNet的学习型模型。

### 7.3 姿态估计与ROI回退

YOLO11 Pose输出每个人的bounding box、17个关键点和置信度。第一遍使用全帧；如果罚球者太小、覆盖不足或没有合格候选，则裁切 lower-left/lower-right 或 goal-band 区域，以960px重新推理，并将坐标映射回原帧。

来源：YOLO模型和输出格式来自Ultralytics；何时回退、ROI位置、去重和坐标还原由本项目实现。

### 7.4 短窗口人物跟踪

对每条已有track，用最近两个中心估计常速度：

\[
\hat c_t=c_{t-1}+v\Delta t
\]

候选匹配成本为：

\[
C=\frac{\lVert c_{det}-\hat c_t\rVert}{\max(s_{track},s_{det})}
+0.30\left|\log\frac{s_{det}}{s_{track}}\right|
\]

超过中心距离门限或尺度比门限的候选被拒绝，其余按成本贪心匹配。

来源：常速度预测和最近邻关联属于通用tracking思想；该成本函数、门限和确定性实现是项目代码。它没有使用SoccerNet、StrongSORT、ByteTrack或Re-ID。

### 7.5 罚球者与触球帧

罚球者候选评分综合：

- 姿态覆盖率；
- track内部连续性；
- 髋部位移；
- 下肢运动强度；
- 触球附近脚踝运动峰值；
- 人体框尺度和画面位置；
- 动作峰值在镜头中的时间合理性。

触球帧候选使用髋部相对脚踝运动：

\[
r^L_t=a^L_t-h_t,\quad r^R_t=a^R_t-h_t
\]

\[
m_t=\max(\lVert r^L_t-r^L_{t-1}\rVert,
\lVert r^R_t-r^R_{t-1}\rVert)/body\_height
\]

在镜头的有效live窗口内寻找最大或显著局部峰值，并排除过晚的射后恢复动作。历史触球帧CSV来自AI视觉估计；它是步数、守门员和球轨迹共同使用的时间锚点，因此其未验证状态也影响下游结果。

### 7.6 左脚/右脚

在触球帧前后小窗口内，先减去髋部中心，分别计算左右脚踝的逐帧2D位移。最大短期位移较大的一侧判为射门脚：

\[
foot=\arg\max_{q\in\{L,R\}}
\max_t\lVert r^q_t-r^q_{t-1}\rVert
\]

来源：向量距离是NumPy计算；髋部去平移、时间窗口和分类规则是项目设计。当前没有充分的真人复核记录，应称为待验证的motion-assisted classification。

### 7.7 助跑起点与步数

髋部中心速度按身体尺度归一化。起动阈值为：

\[
\tau=clip(Q_{0.25}(v)+2.5\,MAD(v),0.006,0.025)
\]

检测静止到持续运动的第一个转折，并保留约0.35秒pre-roll，以避免遗漏第一步。

步数信号将左右脚踝相对位置投影到助跑方向 \(u\)，再除以腿长尺度 \(b_t\)：

\[
s_t=\frac{[(a^L_t-h_t)-(a^R_t-h_t)]\cdot u}{b_t}
\]

对 \(s_t\) 与 \(-s_t\) 分别使用SciPy `find_peaks`，左右极值对应左右脚交替落地候选。项目规则还包括：

- 起点时已经落地的脚不重复计数；
- 保留触球前明显的支撑脚endpoint；
- 抑制触球前0.18秒内射门脚的最后摆动峰，避免把射门动作算成额外一步；
- 合并0.10秒内的重复峰；
- 检查左右交替和最短合理步间隔；
- coverage、关键点置信度、信号强度和交替质量共同决定 `review_required`。

`find_peaks`的函数来源：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>

峰值之外的运动学信号、阈值组合和点球步数定义是项目实现，不来自SciPy。

### 7.8 守门员身份和跨ID拼接

守门员候选综合：

- 触球前/后的轨迹覆盖率；
- 髋部关键点质量；
- 触球后的最大身体尺度位移；
- 与已知罚球者的距离；
- 更靠近图像上方/球门线的先验；
- 人物框较小的远处球员先验；
- 脚下区域的草地HSV比例；
- 排除触球前高速移动的裁判/其他球员。

快速扑救可能导致tracker ID断裂，所以从球门线附近种子开始，用中心距离和尺度变化逐帧向前/向后拼接。该逻辑称为 `goal_line_seed_plus_temporal_stitching`，是本项目专用方法。

### 7.9 守门员镜头运动校正

步骤如下：

1. 灰度化并缩小画面；
2. 使用 `goodFeaturesToTrack`寻找Shi-Tomasi角点；
3. 用Lucas-Kanade稀疏光流跨帧跟踪角点；
4. 使用RANSAC估计 `estimateAffinePartial2D`；
5. 累积变换并将守门员中心映射回首帧坐标。

OpenCV官方光流说明：<https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html>

OpenCV提供算法实现；将其用于守门员轨迹稳定、记录成功率并在低于0.75时触发QA，是项目设计。

### 7.10 守门员罚前活动与扑救分类

对轨迹 `(t,x,y,body_height)`：

\[
v_t^{body}=\frac{\sqrt{(\Delta x)^2+(\Delta y)^2}/\Delta t}
{(b_t+b_{t-1})/2}
\]

采用中位速度减少异常值影响；横向范围为：

\[
R_x=\frac{Q_{0.90}(x)-Q_{0.10}(x)}{median(body\_height)}
\]

再将活动划分为 `static`、`moderate_jitter` 和 `high_active`。

扑救方向和高度取相对于触球前baseline的最大归一化位移点：

- 横向小于 -0.22 body height：`image_left`；
- 横向大于 +0.22：`image_right`；
- 否则：`center`；
- 图像y向上超过0.32 body height：`high`；
- 图像y向下超过0.18：`low`；
- 否则：`mid`。

这些阈值是项目操作定义，不是足球规则，也不是SoccerNet标准。

### 7.11 足球候选与轨迹

使用YOLO11 object detection的COCO class 32 `sports ball`，在触球前3帧到触球后32帧附近运行高分辨率检测。对每个候选计算：

\[
A=1.4G+1.1W+0.35R
\]

其中：

- \(G\)：候选周围patch的草地绿色比例；
- \(W\)：候选框内部的白色比例；
- \(R\)：检测框的长宽圆度。

起始路径分数综合YOLO confidence、外观分、与触球脚踝的距离、与球门的初始距离和时间偏差。

后续每帧候选需要满足：

- 单帧速度不超过门限；
- 与常速度预测位置的残差在自适应容差内；
- 触球后不能长时间静止；
- 短暂漏检最多保持若干帧；
- 最终路径总体朝球门前进且方向一致。

beam search只是一种搜索策略；本项目的评分函数和门限是自定义的，没有来自SoccerNet baseline。

### 7.12 Outcome

`goal/no_goal`来自人工/官方观察标签，而不是由最后球心是否落入一个矩形自动决定。几何位置只用于target-zone辅助解释。

这样避免将以下情况错误合并：

- 轨迹在球门框附近但被扑出；
- 球击中门柱/横梁；
- 球检测在网前丢失；
- 透视误差让像素终点看似在门内。

来源：结果标签本身是人工观察；把结果与轨迹几何解耦是项目设计。

---

## 8. 老师问“这段代码从哪里来”时的标准回答

### 问：YOLO代码是谁写的？

> YOLO模型结构、权重加载和predict接口来自Ultralytics；我们没有重新实现或训练YOLO。我们编写的是如何把它的box和COCO-17关键点转化为点球特征的后处理流程。

### 问：姿态估计是否来自OpenCV Bootcamp？

> Bootcamp提供了姿态估计的概念和OpenPose教学示例。本项目采用相同任务思想，但实际模型是Ultralytics YOLO11 Pose，不是OpenPose，也没有复制Bootcamp的OpenPose推理代码。

### 问：跟踪是不是SoccerNet代码？

> 不是。SoccerNet使用标准MOT和更完整的Re-ID/StrongSORT类流程；本项目为短点球片段实现了一个确定性的中心距离和尺度匹配器，并为守门员增加了点球场景的跨ID拼接。

### 问：为什么可以引用SoccerNet？

> SoccerNet用于证明这些子任务是足球视频分析中的标准研究问题，并帮助解释我们的任务分解。它是related work和benchmark reference，不是直接代码依赖。

### 问：足球轨迹是不是物理弹道？

> 不是。它是2D图像坐标中的检测关联轨迹，使用一阶速度连续性和朝球门方向约束。速度单位是像素每秒，不能解释为真实球速。

### 问：哪一部分是你们真正的贡献？

> 贡献不在训练一个新的foundation model，而在于把通用CV模块组合成可审计的点球分析流程，并定义了点球专用的时间锚点、步数相位信号、守门员动作指标、足球轨迹评分及人工复核机制。

---

## 9. 当前来源说明的薄弱点与修正建议

### 9.1 没有Git历史

当前目录不是Git仓库，无法展示每个算法何时加入、由谁修改、为什么修改。建议在提交前初始化版本控制并从现在开始保留commit；不要伪造过去的历史。

### 9.2 主notebook混有大量探索性cell

主notebook前150个cell包含单个球员试验、变量覆盖、手工帧号和多种尚未采用的计步尝试。最终可重复流程从第150个cell之后开始。展示时应说明：

- 前半部分是exploratory analysis；
- 后半部分调用正式 `src/` 与 `scripts/`；
- 最终结果以processed CSV和diagnostics JSON为准。

更理想的提交形式是另建一个只保留数据读取、最终方法、结果图和结论的clean notebook。

### 9.3 `src/pipeline.py`可能引起误解

该文件含合成数据和sample trajectories，应标记为legacy或移入archive。否则老师可能误以为最终结果是随机生成的。

### 9.4 “homography”和“trajectory”容易被过度表述

- 当前homography类存在，但核心结果没有完成统一球场坐标标定。
- 当前trajectory是2D image-space tracking，不是3D或物理弹道。
- 守门员使用的是背景仿射稳定，不等同于SoccerNet完整camera calibration。

### 9.5 库测试不等于模型准确率

`tests/`验证了公式、边界条件和输出schema，但没有给出真实视频的pose AP、tracking HOTA或ball-centre error。答辩时应把“software unit tests”和“empirical model validation”分开。

---

## 10. 建议写入报告的 Method Attribution 段落

### 英文版本

> The workflow combines standard computer-vision components with project-specific penalty-kick heuristics. Video decoding, colour conversion, feature tracking, affine/homographic transformations, and drawing utilities are provided by OpenCV. Human keypoints and sports-ball proposals are obtained from COCO-pretrained Ultralytics YOLO11 pose and detection models. SciPy's `find_peaks` is used only to locate extrema in the derived ankle-phase signal. All penalty-specific post-processing - including kicker and goalkeeper selection, contact-frame anchoring, run-up onset, step definition, support-foot handling, goalkeeper activity categories, ball-path scoring, review flags, and annotation merging - is implemented within this project. SoccerNet is used as related work for camera-shot segmentation, multi-object tracking, camera calibration, ball-action spotting, and game-state reconstruction; no SoccerNet package or baseline code is imported by the current pipeline.

### 中文版本

> 本研究将标准计算机视觉组件与点球场景专用规则结合。视频解码、颜色转换、特征点跟踪、仿射/单应性变换及图像绘制由OpenCV提供；人体关键点和足球候选来自COCO预训练的Ultralytics YOLO11姿态与目标检测模型；SciPy的`find_peaks`仅用于定位项目所构造脚踝相位信号中的极值。罚球者与守门员选择、触球时间锚点、助跑起点、步数定义、支撑脚处理、守门员活动分类、足球路径评分、人工复核标记和标注合并均由本项目实现。SoccerNet作为镜头分割、多目标跟踪、相机标定、球动作定位和比赛状态重建的相关研究参照；当前流程没有导入SoccerNet软件包或baseline代码。

---

## 11. 推荐引用清单

1. Gopal, V., & Lim, T. W. *Data Analytics for Sense-Making*, Chapter 9, pp. 201-216.
2. OpenCV. “Free OpenCV Bootcamp.” <https://opencv.org/opencv-free-course/>
3. OpenCV. “Lucas-Kanade Optical Flow in OpenCV.” <https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html>
4. OpenCV. “Feature Matching + Homography.” <https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html>
5. OpenCV. “Thresholding Operations using inRange.” <https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html>
6. Ultralytics. “Pose Estimation with Ultralytics YOLO.” <https://docs.ultralytics.com/tasks/pose/>
7. Ultralytics. “YOLO11.” <https://docs.ultralytics.com/models/yolo11/>
8. SciPy. “`scipy.signal.find_peaks`.” <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>
9. Cioppa, A., et al. “SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos.” CVPR Workshops, 2022. Official code: <https://github.com/SoccerNet/sn-tracking>
10. Magera, F., et al. “A Universal Protocol to Benchmark Camera Calibration for Sports.” CVPR Workshops, 2024. Official code: <https://github.com/SoccerNet/sn-calibration>
11. Somers, V., et al. “SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap.” CVPR Workshops, 2024. Official code: <https://github.com/SoccerNet/sn-gamestate>

---

## 12. 最终来源层级图

```text
课程教材 / OpenCV Bootcamp
  提供：CV概念、预处理、tracking、pose、geometry
                    |
                    v
开源软件库
  OpenCV + Ultralytics YOLO + SciPy + NumPy/Pandas
  提供：底层API、预训练权重、数值与峰值工具
                    |
                    v
本项目点球专用实现
  身份选择 -> 触球帧 -> 左右脚 -> 步数
  守门员稳定/动作 -> 足球路径 -> Outcome -> QA
                    |
                    v
SoccerNet对照
  用于：related work、任务分解、未来标准化评估
  不用于：直接导入当前项目代码
```

这一层级是答辩中最需要讲清楚的内容：**使用开源模型并不等于复制完整研究方案；项目贡献在于问题定义、特征工程、场景约束、可追溯输出和人工复核设计。**
