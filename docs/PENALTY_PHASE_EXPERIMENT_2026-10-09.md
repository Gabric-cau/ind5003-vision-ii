# 点球动作分类与步数优化实验

日期：2026-10-09。当前范围是39条已有点球剪辑。本文描述已经实现并运行的代码，
不是未来计划，也不是人工标注报告。完整逐视频摘要见
`outputs/phase_optimization_2026-10-09/review/report.html`。

## 先明确结论和证据等级

本轮可交付的是“阶段＋动作描述＋步伐候选＋不确定性”，不能宣称已得到真实落脚步数。
31条有暂定阶段，8条不强行分类。独立球—人关联20条较强证据、12条暂定、7条弃权；
32条有关联中，Kramarić只有很短的轨迹，所以无法解释完整助跑。

98项选定测试通过，78条两版结果通过结构检查，39条完整播放副本解码核验通过。
这些数字是计算完整性、工程测试和质检，不是动作准确率。
没有独立逐帧人工真值；Cuadrado的“约5步，4–6步可接受”是用户真实提供的粗略意见，
仅适用于这一例，不能变成精确5步标签，也不能自动扩展到所有人。

## 为什么不是简单把10步分成5步调整和5步助跑

原流程可能选错人、把摆腿最大值当触球，或把触球后动作继续计数。
Cuadrado旧10个候选中的一些本来就在新的触球候选之后，不能全解释成赛前小步。
因此先独立检查镜头、球和人物，再讨论阶段。旧主表和旧缓存全部保留作对照。

本轮AI检查39张旧版QA图，发现11条轨迹明显落在门将或裁判；具体证据帧见
`docs/penalty_phase_methodology_2026-10-09.json`。这是AI质检，不是人工标签。
第一轮31条非弃权球—人关联的抽样画面均已AI检查，另单独检查了Kovačić重选镜头；
不能据此保证每一帧身份都正确。

## 分类定义：阶段和属性分开

| 层级 | 输出 | 如何解释 |
|---|---|---|
| 时间阶段 | preparation_adjustment | 正式接近之前；可能是站立，也可能有调整，不表示整段都在走 |
| 时间阶段 | initiation | 推断起点后的短窗口；不是直接测得的加速度 |
| 时间阶段 | approach | 向触球附近连续接近；允许其中有短暂停顿、小步或弧线 |
| 时间阶段 | strike | 估计触球附近窗口；可能包含支撑脚、摆腿和真正触球 |
| 时间阶段 | recovery | 触球候选之后的随动恢复，不混入名义助跑计数 |
| 时间阶段 | unknown | 遮挡、缺帧、身份或触球不足，保留未判定 |
| 可并存属性 | 横移、弯曲、绕圈状投影 | 二维路线描述，不是已标定的地面几何 |
| 可并存属性 | 提速、暂停、密集低推进量 | 相对运动描述，不是米/秒冲刺或真实步长/步频 |

不先给每个球员指定固定点球风格，再反推视频。只描述本次罚球；“绕圈型球员”或
“门将依赖型策略”需要更多可观察证据。当前未检出足够的横移/完整绕圈证据，不凑分类数量。

## 实际流程、方法和代码来源

1. **原始证据与镜头**：复用已有YOLO11n Pose逐帧缓存，保留原视频绝对帧号。
   `scripts/rescue_phase_shots.py` 对Kovačić检测到旧[0,39)不是实际罚球镜头，自动尝试
   后续镜头[40,133)，独立触球候选为77。旧姿态缓存中仍有116帧旧踢球估计，主实验明确不使用它。
2. **独立触球与人物关联**：`src/kick_contact.py` 用YOLO物体检测的sports ball候选，
   背景补偿后的相对静止球位置、所有人物踝部接近、连续离开球位的检测证据排序。
   比较不同人物，拒绝弱证据或明显歧义。不是要求“看不到球动就丢视频”，而是缺证据时不武断指定触球。
   近球脚踝可能是支撑脚，所以输出启发式窗口；它不是统计置信区间或精确触球标签。
3. **镜头补偿**：`src/motion_stabilization.py` 在人物框外检测角点，调用OpenCV
   `goodFeaturesToTrack`、`calcOpticalFlowPyrLK`、正反向检验和`estimateAffinePartial2D`/RANSAC。
   将轨迹映射到所选镜头首帧。任何失败都会使后续参考变换为null，不静默用单位矩阵。
   不完整补偿时，阶段使用原图坐标并明确警告；相机检查通过仍不等于地面标定。
4. **阶段识别**：`src/penalty_phases.py` 过滤髋踝低置信度，短缺口插值、长缺口保留unknown，
   对髋部轨迹平滑、求速度，结合踝部活动和终端接近方向找连续运动段。
   新增0.4秒窗口净推进与方向一致性检查，避免仅用瞬时速度把站立晃动吸入助跑。
   保留`speed_only_onset_frame`与推进证据，便于解释起点怎么改变。慢动作/透视不足时仍警告并保留暂定，
   不把规则无法确认等同于人没走。
5. **步伐候选**：`src/phase_step_events.py` 对左右踝相对差向量按人物框高归一化，
   用PCA选主运动轴，中值平滑后调用SciPy `find_peaks` 获取正负极值。
   三档显著度0.025、0.04、0.06并列；同极性最小间隔0.22秒，交叉极值去重间隔0.12秒。
   未把未知落脚腿强行写成left/right，未添加缺失的真实接触，也未按用户“5步”拟合参数。
6. **分类后计数**：准备候选、名义助跑候选、触球窗口歧义候选分列。
   早/晚触球边界和三档峰谷参数给出敏感性，不是统计置信区间。不要把所有列相加。
   事件可来自真实脚步、摆腿、原地调整或姿态抖动，需看原片确认。
7. **小步/密集动作**：由于落脚腿未知，不能直接用每只脚的“短位移”分支冒充碎步识别。
   本轮改用相邻候选之间的稳定化髋部推进量和时间间隔作补充；连续至少3个间隔满足
   推进0–0.20个框高、间隔不超过0.32秒时，给局部“密集低推进量动作候选”标签。
   这些阈值是披露的工程选择，不是经过训练验证的生物力学界限。小步本身可能是真实助跑，不能简单删除。
8. **大姿态模型消融**：`src/pose_refinement.py` 在自动关联人物框附近裁剪，用已有YOLO11m Pose重估，
   映射回原图。相邻人物匹配歧义时拒绝，不能转而追踪旧门将。30条可用对照、9条弃权。
   必需髋踝样本2661→2662，覆盖收益极小；最终30个实际对照中19个候选数变化、8个差至少2、
   5个起点差超过4帧、8个标签集合改变。没有已验证改进，因此不自动升级默认结果。
9. **本地复核台**：`scripts/templates/penalty_phase_review.html` 加完整H.264原片副本，
   可逐帧、慢放、显示人物框、按阶段/事件跳转、切换姿态版本、盲看。只读，不上传，不伪造复核记录。

第2、4、5、6、7步的组合逻辑和阈值是本项目新编写的启发式实现，不是声称从课程或SoccerNet复制。
第三方库提供底层检测、光流和信号处理能力；研究论文仅作为概念依据。

主要资料：

- [Ultralytics Pose官方文档](https://docs.ultralytics.com/tasks/pose/)
- [OpenCV光流官方教程](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html)
- [SciPy find_peaks官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)
- [Body Pose Estimation Integrated With Notational Analysis, 2022](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2022.818556/full)
  将姿态与观察分析结合；本项目只借鉴“动作与策略分开”的分析思路，未复制其OpenPose流程，未借用其准确率。
- 更多原始资料及SoccerNet角色识别概念对照，见本目录`penalty_phase_methodology_2026-10-09.json`。

## 结果位置

| 文件夹/文件 | 用途 |
|---|---|
| `contact_probe/` | 第一轮39条球—人关联，19/12/8；冻结保留，不包括后续Kovačić救援 |
| `shot_rescue/` | Kovačić重新选镜头、姿态和触球覆盖证据；加入最终主实验后为20/12/7 |
| `camera_motion/` | 参考变换、逐对质量、断链标记 |
| `classified/` | 最终默认候选阶段，39 JSON及QA图，summary.json汇总 |
| `refined_pose/` | YOLO11m裁剪对照的原始检测、覆盖与来源 |
| `classified_refined/` | 对照分类；未增强条目会明确保留原姿态来源，不计入30条真实消融 |
| `model_comparison.json` | 两种姿态的候选数、起点、标签分歧 |
| `validation.json` | 98项选定测试、两版结果结构校验 |
| `review/` | 完整视频、只读复核页、晨间报告、媒体解码校验、服务器日志 |

以上均位于`outputs/phase_optimization_2026-10-09/`。原`data/processed/`、原视频和旧notebook不覆盖。

## 如何复现

已有有效Python环境下：

```powershell
python scripts/probe_kick_contact.py --all
python scripts/rescue_phase_shots.py
python scripts/refine_phase_pose.py --all --reuse
python scripts/run_penalty_phase_experiment.py
python scripts/run_penalty_phase_experiment.py --output-dir outputs/phase_optimization_2026-10-09/classified_refined --refined-dir outputs/phase_optimization_2026-10-09/refined_pose
python scripts/build_penalty_phase_review.py
python scripts/validate_penalty_phase_experiment.py
python scripts/make_penalty_phase_report.py
python scripts/serve_penalty_phase_review.py --port 8882
```

服务仅监听127.0.0.1。浏览器打开`http://127.0.0.1:8882/report.html`或`index.html`。
如果只更新页面和结果、不重新转码，构建加`--data-only`。媒体不会在校验不匹配时静默覆盖。
触球/救援/增强脚本的重跑输出策略不同，运行前查看`--help`；已有结果不应被当成人工标签。

本机实际使用`C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe`（Python3.14.7），
NumPy2.4.6、SciPy1.17.1、OpenCV4.13.0.92；Ultralytics8.4.137、Torch2.13.0 CPU由
项目`.venv\Lib\site-packages`追加导入路径提供。不能把该目录放在数值库之前；某些旧依赖目录
存在访问限制，不应绕过。实际核心版本另列于`requirements-phase-experiment.txt`，不是全项目的干净安装锁文件。

需要模型的脚本，在本机可按已验证的导入顺序运行，例如：

```powershell
& C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe -X utf8 -c "import sys,runpy; sys.path.append(r'C:\Users\30605\PycharmProjects\JupyterProject\.venv\Lib\site-packages'); sys.argv=['scripts/probe_kick_contact.py','--all']; runpy.run_path(sys.argv[0],run_name='__main__')"
```

已有FFmpeg位于`outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`。
视频构建脚本允许`--ffmpeg`指定其他安装。没有修改系统Python或替换旧requirements。
未验证全新机器的一键安装；所验证的是本机现有环境下的运行与检查。

## 教师可能追问的边界

- **为什么清楚的画面也会弃权？** 检测器可能混淆球与罚球点、漏掉运动模糊中的球、断开人物轨迹，
  或候选帧其实已经射门结束。列明具体证据缺口，不能用“肉眼不清楚”搪塞。
- **为什么不用mask硬筛？** HSV二值草地区域不能独立说明谁踢球、何时触球。本轮只用人物几何区域排除背景运动估计干扰，
  没有开启HSV草地门槛。也没有把“肉眼看不到球运动”直接判成无效罚球。
- **为什么不是homography后的真实步长？** 当前没有可靠逐镜头球场标定；髋部不在草地平面，球也可能腾空，
  不能把平面单应变换应用后的坐标直接解释成三维人体/球真实位置。
- **如何证明优化有效？** 已证明若干错误角色/镜头的候选关联改变，已降低一些无证据强行输出，
  但尚未证明步数或风格准确率提高。应盲看标注后评估人物正确率、触球时差、阶段边界误差、
  真实落脚事件precision/recall与总数MAE，分别报告弃权率和条件准确率。
- **为什么不把19/30模型分歧择优调成用户想要的？** 没有真值就不知道谁更准；选择符合预期的计数会造成确认偏差。
- **后续最有价值的工作？** 先确定准备步、迈步、支撑脚、原地跳步的计数定义，标注少量但独立的落脚帧和阶段边界；
  修复长轨迹拼接与球运动模糊，再做留出验证。不是继续堆模型或仅让输出数量变多。
