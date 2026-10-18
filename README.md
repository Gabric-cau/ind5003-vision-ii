# IND5003 Vision II：点球视频分析

分析课程提供的四场 2018 年世界杯点球大战，共 39 次罚球：哥伦比亚对英格兰 10 次、克罗地亚对丹麦 10 次、西班牙对俄罗斯 9 次、俄罗斯对克罗地亚 10 次。程序提取触球时刻、射门脚别、可见助跑步数、门将动作和球的位置，保存逐帧证据及结果表。

代码基于小组原项目继续修改。主结果来自 10 月 13 日的实验；10 月 18 日修复了落地候选的支撑记录与配对逻辑，用相同的逐帧输入完成回放。

## 结果文件

| 文件 | 内容 |
|---|---|
| [报告 PDF](outputs/fresh_course_experiment/report_v4_continuous_shoe/REPORT_FRESH_REVIEW_DRAFT.pdf) / [Word](outputs/fresh_course_experiment/report_v4_continuous_shoe/REPORT_FRESH_REVIEW_DRAFT.docx) | 10 月 13 日的九页审阅稿，未包含本轮回放 |
| [主结果表](outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.csv) / [JSON](outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.json) | 39 次罚球的特征及来源记录 |
| [脚别补充表](outputs/fresh_course_experiment/shoe_approach_foot_supplement_v2/foot_candidates_39.csv) | 连续推理后的脚别结果，增加 Dzagoev 一段 |
| [落地修复对照](outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/comparison_39.csv) / [实验说明](docs/SHOE_LANDING_REPLAY_2026-10-18.md) | 同一批输入，比较完整支撑记录与旧版三帧截断 |
| [主结果查看页](outputs/fresh_course_experiment/review_v3_shoe_fusion_ui3/index.html) / [10 月 13 日助跑对照](outputs/fresh_course_experiment/shoe_approach_analysis_v2/review_v2/index.html) | 下载仓库后在浏览器打开，查看主结果和修复前的叠图 |
| [人工复核页](outputs/fresh_course_experiment/shoe_approach_analysis_v2/blind_review/index.html) | 隐藏预测的原视频播放器，需另备课程片段 |

![Kane 片段的姿态、球和触球区间检测](outputs/fresh_course_experiment/review_v1/evidence/col_eng_p02_kane_eng9_evidence.jpg)

## 程序使用的方法

| 环节 | 实现 |
|---|---|
| 视频准备 | FFmpeg 切片，OpenCV 解码；记录原录像帧号、片段偏移和 SHA256 |
| 人物姿态与身份 | YOLO11m-pose 提取 17 个关键点；按位置、框大小和运动连续性关联人物，结合球场与球门位置区分罚球者、门将 |
| 足球检测与触球时刻 | YOLO11n、YOLO11m 分别检测足球；追踪静止球到离开罚球点的过程，比较两路检测的身份和时间区间 |
| 脚别与助跑 | MediaPipe Pose Landmarker Heavy 提取 33 个关键点；正常、镜像两路推理检查摆腿、脚跟、脚尖和近球证据；SciPy 提取旧版脚踝极值 |
| 相机运动与地面投影 | OpenCV 光流和 RANSAC 估计背景运动；球场白线、球门几何和单应矩阵生成地面投影点 |
| 门将动作 | 目标关联、姿态位移和动作阶段规则；SAM2-t 用于局部门将轮廓试验 |
| 赛果 | 球轨迹规则；OpenCV 提取转播比分板的红绿圆形图标，按顺序和稳定性判断结果 |
| 汇总与检查 | NumPy、pandas、Matplotlib；pytest 检查代码行为、坐标变换、来源对应和冲突处理 |

模型使用已有权重进行推理，项目没有训练或微调模型。比分板处理使用图标几何和颜色。SAM2 的轮廓试验单独保存，完整扑救高度尚未测出。具体参数见[主实验设置](docs/FRESH_EXPERIMENT_PROTOCOL.md)和[连续助跑实验](docs/SHOE_APPROACH_RESULTS_2026-10-13.md)。

## 当前指标

以下数量表示有候选输出的罚球数。覆盖率以 39 次罚球为分母。独立人工复核目前为 0 段，脚别准确率、真实步数误差和门将动作准确率尚未测得。

| 项目 | 当前输出 | 覆盖率或说明 |
|---|---:|---|
| 触球时刻 | 33/39 | 84.6% |
| 射门脚别 | 14/39 | 35.9%；主表 13 段，补充表增加 1 段；10 右、4 左 |
| 旧版助跑计数 | 32/39 | 82.1%；统计脚踝极值，待核对落地事件 |
| 连续助跑落地候选 | 12 段，共 22 个 | 修复前为 7 段、12 个；使用相同的 32 段输入区间 |
| 门将触球前移动 | 29/39 | 74.4% |
| 门将扑救方向 | 23/39 | 59.0%；按画面左、右、中间记录 |
| 球轨迹 | 33/39 | 84.6%；保留观测点与跟踪中断记录 |
| 地面投影 | 32/39 | 226 个触球前地面候选点；完整俯视路线待完成 |
| 完整扑救高度 | 0/39 | Brozovic 有 1 段带条件的局部图像高度信号 |
| 球的最终位置 | 0/39 | 14 段遮挡、8 段镜头切换、17 段未知 |
| 自动慢动作判断 | 0/320 个镜头区间 | 跨镜头动作对齐尚未建立 |
| 比分板赛果 | 35/39 | 与 FIFA 记录一致 35/35 |
| 球轨迹赛果 | 10/39 | 与 FIFA 记录一致 9/10 |
| 融合赛果 | 35/39 | 与 FIFA 记录一致 35/35 |

连续助跑的 39 段中，12 段有落地候选、15 段未检出、5 段关键点左右通道不稳、7 段没有可用输入区间。未检出表示当前规则未找到合格候选，实际步数仍待核对。

赛果一致率是在参与开发的这批视频上计算的，尚无独立测试集。比分板和融合赛果各覆盖 35 段，两组事件有所不同：Koke 的球轨迹与比分板冲突，融合结果保留未知；Rakitic 的比分板缺失，由球轨迹规则补入。数据见[主结果表](outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.csv)；本轮落地回放的[人工评估状态](outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/evaluation_pending.json)仍为待标注。

## 主要问题与后续工作

旧程序满足三帧低运动条件后就截断支撑记录，导致正常与镜像两路起点相差一帧时漏配。10 月 18 日的修复保留同一阶段的完整支撑，并检查共同持续时间。原有 12 个候选全部保留，增加 10 个。[回放对照](docs/SHOE_LANDING_REPLAY_2026-10-18.md)记录了每段变化。

Rashford 的旧脚踝计数为 15，鞋部候选由 3 个变为 6 个。新增候选中仍有仅约 0.067 秒的短支撑，空中变向也可能满足低运动条件。两套结果需要逐帧标注来比较误差，主表的旧计数保留原值。

脚别覆盖从 8 段增加到 13 段，再增加到 14 段，剩余 25 段未知。新增的 Dzagoev 右脚候选依赖第 262 帧的近球证据；Kane 仍然未知。后续先核对原录像里的触球区间、下肢遮挡和人物身份，再针对失败片段调整裁剪与跟踪。

地面投影和门将高度都依赖相机几何。目前地面标定在 609 个采样帧中接受了 521 帧，仍需核对投影误差与时间连续性。门将高度缺少完整落地阶段和稳定地面参考。下一步按完整动作重新检查镜头和人物连续性，跨镜头结果分别保存。

评估先由两名同学独立标注原课程视频，记录可见助跑区间、落地帧和触球脚。预测已冻结，[复核页](outputs/fresh_course_experiment/shoe_approach_analysis_v2/blind_review/index.html)隐藏自动结果，本轮使用[新版标注模板](outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/human_reference_template.json)。[评估脚本](scripts/evaluate_shoe_landings.py)保留标注分歧。标注完成后计算脚别一致率、落地事件精确率与召回率、可见区间步数 MAE，再比较方法改动。门将方向需另表标注，当前评估脚本尚未覆盖。

## 运行

结果表、报告和检测图片可直接查看。重生成统计图：

```sh
git clone https://github.com/Gabric-cau/ind5003-vision-ii.git
cd ind5003-vision-ii
python3 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python -m src.fresh_eda \
  --bundle-dir outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion \
  --output-dir outputs/my_eda
```

Windows 激活命令为 `.venv-analysis\Scripts\Activate.ps1`。统计图写入 `outputs/my_eda`，同名文件会被覆盖。

```sh
python -m pip install -r requirements-experiments.txt
python -m pytest -q
```

重新提取特征需另备原视频、模型权重和 FFmpeg。推理依赖分别在 `requirements-cuda.txt` 与 `requirements-pose33.txt`，运行入口在 `scripts/run_fresh_*`。代码模块位于 `src/`，配置位于 `configs/`，当前结果位于 `outputs/fresh_course_experiment/`。

## 来源

主结果及连续助跑实验使用原课程的四份录像。早先下载的 FIFA 高清剪辑作为[单独三球对照](docs/HD_PILOT_RESULTS_2026-10-13.md)保存，未进入这两项实验；素材记录见[视频来源](docs/ADDITIONAL_VIDEO_SOURCES.md)。

原小组仓库：[RX422040/vision-ii-penalty-analysis](https://github.com/RX422040/vision-ii-penalty-analysis)。修改基于原项目，代码与文档更新使用了 AI 辅助。报告仍为审阅稿，课程提交由小组另行完成。
