# Vision II 来源纠正、Mask / Edge / Homography 优化记录

日期：2026-10-08。本文取代旧报告中把AI候选称为“人工确认真值”的表述。

## 1. 人工数据到底来自哪里

用户明确表示没有做人工复核。检查此前任务的实际文件创建记录后，确认：

- `data/annotations/penalty_step_manual_reviews.csv` 由助手在此前任务中写入，包含助手/代理视觉审阅的候选计数和落脚帧；并非用户上传或真人复核。
- 创建该文件的历史记录时间为2026-09-01 16:59 UTC，即新加坡2026-09-02 00:59；随后记录还有助手修改两条计数的操作。这是文件作者证据，不是数值正确性证据。
- `penalty_kick_frames.csv` 同样由助手写入，并错误地用了 `manual_step_review` 来源名。
- `penalty_goalkeeper_track_reviews.csv` 的36条种子身份也是助手任务中写入的视觉判断，没有真人签署记录。
- 合并脚本曾无条件写入 `annotation_method=manual_frame_review`，把文件名和助手审阅误当作真人复核。这个设计与此前的汇报措辞都是错误的。
- `penalty_outcomes.csv` 有外部比赛结果来源与URL，来源性质与上述动作标签不同；保留其原始引文，但不能用赛果正确来证明轨迹正确。本轮没有重新逐条核验外部赛果。

目前独立真人确认的步数参考为 **0条**。旧4/35完全一致、12/35相差不超过1步、MAE 2.66、RMSE 3.38等数值，是自动算法与AI候选的分歧，不能作为模型准确性，已撤回相应准确率结论。

## 2. 已经落实的来源修复

1. 三种动作标注追加 `annotator_type=ai_assisted`、`human_verified=false`、空复核者/日期和来源说明；原计数与帧号保留供诊断。
2. 合并脚本保留 `candidate_step_count` / `candidate_landing_frames`；没有完整真人记录时，`confirmed_step_count`、`confirmed_landing_frames`、自动/真人一致字段为空。
3. 真正的人类标注须有 `annotator_type=human`、`human_verified=true`、`reviewer_id`、合法的 `reviewed_at`。这些字段必须来自实际复核，不能通过批量填充绕过。
4. 守门员的 `goalkeeper_identity_reviewed` 改为false；守门员与球轨迹输出追加未验证输入标识。数值特征和QA通过状态没有被伪装成真人结论。
5. 守门员、球轨迹、守门员重推理脚本默认拒绝未验证动作标注。只有显式使用 `--allow-unverified-annotations` 才能复现探索实验。
6. README、方法文档、来源文档和Notebook的确认数据章节已更正；Notebook顶部加入醒目来源说明。
7. 变更前CSV与Notebook备份于 `outputs/archive/provenance_before_2026-10-08/`，可恢复。旧诊断JSON作为历史运行证据保留，不应读取其中的旧reviewed文字作为当前核验状态。

相关代码：`src/annotation_provenance.py`、`scripts/merge_manual_step_reviews.py`、`scripts/correct_annotation_provenance.py`。当前审计结果见 `outputs/annotation_provenance_audit.json`。

## 3. 三种课程方法如何配合

| 方法 | 在本项目中的用途 | 不能由此得到的结论 |
|---|---|---|
| HSV / inRange / 二值Mask | 提取绿色候选区域，限制球场搜索；白色候选辅助球和线条提取；排除人物框后估计背景运动 | 绿色不必然是草地；白色不必然是足球；Mask本身不输出步数 |
| Gaussian + Canny + HoughLinesP | 在球场区域内寻找白线边缘和直线段，作为几何标定候选 | 检出的线段可能是球网或广告，不能自动知道是哪条边线/禁区线 |
| Homography / 透视变换 | 把正确匹配的球场地面坐标映射为俯视坐标，改善地面方向、位移和尺度比较 | 不能从单张变换图恢复足球高度，不能把空中球心或髋部直接视为地面点 |

建议的依赖顺序：镜头选择 → HSV球场候选 → 场内白线边缘 → 语义线段/标志点匹配 → 有效单应性 → 地面轨迹；人体姿态计步保留原始视角的脚部证据，与地面运动共同分析。

不把整个人物先用绿色Mask抹掉再送入Pose模型，否则会破坏模型需要的外观。当前Mask只是候选评分和背景角点选择的条件。

## 4. 本轮新增实现与实际实验

`src/field_masks.py` 包含以下实现：

- 在宽松绿色HSV区间内估计主色调，形成自适应阈值。
- 开运算去小噪声、闭运算连接碎片；选择最大区域并填孔，避免球员与白线造成空洞。不使用凸包扩张到看台。
- 检查区域比例、下部绿色覆盖、横向跨度和宽行数量。质量不足时返回未知，不强行排除候选。
- 以人物框脚下区域的球场覆盖作弱评分，最多减3分，避免硬阈值直接删除真实球员。
- 在背景运动估计中排除人物框和顶端覆盖层，优先使用球场区域。场地不足时回退到排除人物后的背景。
- 生成白色小连通区域；明确标为候选，不替代YOLO检测或人工标签。球飞到球网、看台或空中时不能被绿色Mask硬删除。
- 在原图灰度上做Canny后再限制ROI，避免先抹黑背景制造虚假Mask边缘；随后做Hough线段提案。仍需辨认真实场地线。

### 步数候选消融实验

用原有固定Pose缓存，保持检测结果不变，只改变人物候选的Mask评分：

| 检查项 | 结果 |
|---|---:|
| 点球片段 | 39 |
| 解码的Mask抽样帧 | 1468 / 1468 |
| 通过掩膜形状启发式检查的帧 | 1253 |
| 原有非空计数 | 35 |
| 加Mask后的非空计数 | 35 |
| 罚球者轨迹选择改变 | 0 |

**这次实验没有证明计步改善。** 1253是启发式通过数，不是分割准确数。绿色球衣仍可能造成误检，阈值也可能拒绝真实草地；需独立的场地Mask参考才能测分割IoU。

实验：`scripts/run_mask_ablation.py`。
产物：`outputs/mask_optimization_2026-10-08/comparison.csv`、`summary.json`、`diagnostics.json`、`qa/`下39张对照图。

### 守门员背景Mask消融实验

两次运行使用相同Pose缓存、AI候选触球帧与身份种子，分别关闭/开启背景Mask；两者都显式标为使用未验证输入的探索实验：

- 39条均完成，两组自动QA通过数都是18，待检查都是21。
- 3段的分类发生变化：Pique的方向、Ramos的高度/活动量、Vida的活动量。
- 分类变化不等于更准确。没有真人参考时不能选择看起来更合理的那组当作胜者。

产物：`goalkeeper_baseline.csv/json` 与 `goalkeeper_mask.csv/json`，均在同一实验目录。
代码开关：`scripts/run_goalkeeper_analysis.py --field-mask`。

### 边缘与几何预览

`scripts/render_pitch_geometry_qa.py` 已生成39张“原图—Mask+Canny—Hough候选”对照图，位于实验目录的 `pitch_geometry/`。这是标定候选证据，不是已经完成39段球场标定。

## 5. 4次失败的更重要原因：镜头选择

旧流程只分析第一镜头，遇到第一次明显切镜就停止。这4条的处理区间都是开头的短镜头：

| 点球片段 | 旧处理帧数 | 30fps对应时长 |
|---|---:|---:|
| Croatia–Denmark，Modric | 28 | 0.93秒 |
| Spain–Russia，Golovin | 33 | 1.10秒 |
| Russia–Croatia，Ignashevich | 25 | 0.83秒 |
| Russia–Croatia，Vida | 46 | 1.53秒 |

从原图可见，开头包含罚球者/守门员/观众特写，实际射门全景在后面。旧缓存根本没有分析到所需的完整助跑。单独改变计步峰值阈值或Mask选人评分无法补回这些未提取的姿态。

`scripts/audit_shot_windows.py` 新增了只读镜头诊断，在前12秒内找候选全景；3条出现候选窗口，Vida暂时没有通过候选条件。它也暴露出仅用HSV直方图切镜会漏掉颜色类似的全景→特写转换。

这些窗口是算法提案，尚未作为正式片段重新推理。下一轮需要以镜头完整性、人体全身可见性与触球前后事件共同选窗；不能为了把35/39变39/39就强行输出数字。

四段原始片段见 `data/raw_clips/`。为方便回看，已在 `C:/Users/30605/Downloads/VisionII_failed_four_review/` 生成完整时长的H.264副本，帧数与原片相同，原件未改。

## 6. Homography已修复什么、还缺什么

旧的 `src/homography.py` 只是OpenCV封装；默认没有矩阵时直接返回像素坐标。旧助跑函数却把距离写到 `_m` 字段，存在单位误报风险。

已实现：

- 至少4组有限、不同且整体不共线的匹配点检查；失败时清空旧矩阵，防止误用前一镜头标定。
- 用地面→图像的逆向拟合运行RANSAC，使误差阈值始终是像素，再取逆得到图像→地面变换。
- 输出拟合内点数和像素重投影误差；4点完全拟合不等于独立验证。
- 明确 `destination_unit=arbitrary` 或 `m`；没有明确米制标定时，距离和速度的米制字段为空，另存像素/任意单位结果。
- 拒绝落在投影奇异位置的轨迹点。

尚缺真实球场标志点的语义匹配与独立验证，因此没有把当前39段结果改成米或km/h。建议至少多于4个、分散覆盖所需区域的对应点，用额外点检查；平移、缩放或切镜之后必须更新/验证矩阵。对球员应使用足部接地点；对足球只在球确实处于地面时使用地面H。射出后空中球轨迹仍以图像坐标报告。

## 7. 下一轮优化优先级和验收标准

1. **选对射门窗口**：先修复4段短开头截断；同时检查原来35段是否包含完整助跑，保留原视频绝对帧号与切片偏移。验收不是有数值，而是动作区间覆盖正确。
2. **联动选罚球者和触球帧**：不能只看哪人的脚踝运动最大。结合静止球候选、脚—球距离、球开始持续离开与人体接近过程，输出时间不确定区间；关键点缺失时重推理目标ROI。
3. **计步状态序列**：在原始视角跟踪左右脚，从摆动→减速接地→支撑→离地构造事件，结合关键点置信度、双脚交替与最短持续时间。峰谷只是候选，不能每个峰都加一步。
4. **背景稳定与地面几何**：评估本轮Mask光流，再用通过验证的场地标定改善地面位置。把人体高度、扑救空中动作与地面运动分开。
5. **球追踪**：白色Mask作候选补充，结合高分辨率检测和运动预测；碰门柱、扑救后反弹、出框要作为状态变化，不能把插值当检测。
6. **独立验证**：先由真实复核者覆盖4场比赛和主要失败类型，标记身份、触球帧、助跑起止和落脚帧。按比赛留出测试，调参只用开发部分；小样本应报告逐片误差和区间。无真实标注时只报告覆盖、稳定性、分歧和失败类型。

这轮完成了来源修复、Mask/Edge实现与消融、几何单位修复和镜头失败诊断；未完成新的全流程模型重推理，也没有证据宣称准确率提升。

## 8. 复现与验证

原 `.venv` 指向已不可启动的Python路径。本轮使用本机打包Python 3.12和项目内 `.deps_mask/` 运行CPU实验，未改写旧虚拟环境；依赖见 `requirements-mask.txt`。

在正常Python环境中：

```powershell
python -m pip install -r requirements-mask.txt
python scripts/run_mask_ablation.py
python scripts/audit_shot_windows.py
python scripts/render_pitch_geometry_qa.py
python scripts/run_goalkeeper_analysis.py --allow-unverified-annotations --output-csv outputs/mask_optimization_2026-10-08/goalkeeper_baseline.csv --diagnostics-json outputs/mask_optimization_2026-10-08/goalkeeper_baseline.json
python scripts/run_goalkeeper_analysis.py --allow-unverified-annotations --field-mask --output-csv outputs/mask_optimization_2026-10-08/goalkeeper_mask.csv --diagnostics-json outputs/mask_optimization_2026-10-08/goalkeeper_mask.json
python -m pytest tests -q
```

当前25项测试通过，覆盖现有计步/守门员/球模块及新增来源隔离、Mask回退、人物排除、白色候选、边缘提取、退化标定和米制单位保护。单元测试不是实拍精度评测。

官方方法来源：

- [OpenCV HSV/inRange](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)
- [OpenCV morphology](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [OpenCV Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [OpenCV Hough line transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)
- [OpenCV homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)

颜色范围、形状门槛、3分人物评分权重、线长门槛和回退策略均为本项目的实验设计，尚未通过独立测试集调优验证。
