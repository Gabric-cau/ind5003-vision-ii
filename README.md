# IND5003 Vision II：点球视频分析

分析课程提供的四场 2018 年世界杯点球大战，共 39 次罚球：哥伦比亚对英格兰 10 次、克罗地亚对丹麦 10 次、西班牙对俄罗斯 9 次、俄罗斯对克罗地亚 10 次。程序提取触球时刻、射门脚别、可见助跑步数、门将动作和球的位置，保存逐帧证据及结果表。

代码基于小组原项目继续修改。主结果来自 10 月 13 日的实验；10 月 18 日修复了落地候选的支撑记录与配对逻辑，用相同的逐帧输入完成回放。

## 本地复现当前模型

下载 [v5 完整模型运行包](https://github.com/Gabric-cau/ind5003-vision-ii/releases/tag/v2026.10.28-portable-v5)，解压后进入 `portable_bundle_v5`。模型权重随 Release ZIP 提供；GitHub 自动生成的 Source code ZIP 仅包含源码，不能代替运行包。完整安装、输入范围和验证记录见[复现说明](docs/REPRODUCE_PORTABLE_V5_2026-10-28.md)。

Windows PowerShell，Python 3.11，RTX 2060/4060 等显卡：

```powershell
py -3.11 install.py --torch cu118
.\.venv\Scripts\python.exe run_video.py --video C:\videos\penalty.mp4 --output runs\my_penalty --device cuda:0 --batch-size 1
```

RTX 5060 等较新显卡选择 `--torch cu128` 并确保驱动支持；没有 NVIDIA GPU 的机器选择 `--torch cpu`，运行时使用 `--device cpu`。安装依赖需要联网；安装完成后，推理读取本地视频和权重。

Linux/macOS 的命令使用 `python3.11 install.py --torch cpu` 和 `.venv/bin/python run_video.py ... --device cpu`。Linux NVIDIA 机器可改用 `--torch cu118` 或 `--torch cu128`。

输入应为已经切好的单次罚球 MP4，从射门主镜头开始，包含助跑、触球和可见结果。结果写入 `analysis.json`，最终字段在 `required_outputs` 和 `final_feature_table`，阶段日志在 `logs/`。每次选择新的输出目录。视频、人工答案和账号配置不包含在运行包中。

这个版本发布当前流程修复、最终训练审计和提交材料初稿，不宣称达到 90% 准确率。最新球/门将增广训练没有超过保留基线，因此默认流程没有被替换；严格扑救高度和事件最终球位置仍保留为 unknown。历史脚别参考的部分字段待裁决；下面的历史分数应结合其版本和参考标签阅读。

已在 Dell RTX 2060（6GB、Windows 11）上从两套空虚拟环境完成安装验证，并用一段不带人工答案的视频跑通全部 14 个阶段，用时约 3 分 41 秒。[实机验证记录](docs/DELL_PORTABLE_V5_VALIDATION_2026-10-28.md)记录了运行包哈希、环境和限制；这项检查证明可运行，不是新视频准确率评估。

## 10 月 26 日训练与离线运行

已新增触球、脚别、落地、门将、球类别和镜头分类的监督训练，以及无需人工答案的新视频入口。训练结果、十段完整运行和独立比赛测试分别记录在[训练与部署说明](docs/TRAINING_AND_DEPLOYMENT_2026-10-26.md)。预训练视觉网络保持冻结，训练的是下游分类器；以下主表和审阅稿保留历史版本，不代表最新模型的全部表现。离线包位于 `outputs/offline_deployment_2026-10-26`，具体运行方法见包内 README。

完整 v3 流程已在另一场比赛的 8 段视频上完成 8/8、0 失败。冻结评分和运行时间见[完整流程评估](docs/WORLDCUP_FULL_PIPELINE_EVALUATION_2026-10-27.md)：脚别严格准确率 3/8，赛果与门将定性高度当前均因证据门槛输出 0/8；中位端到端时间 287.15 秒，p95 468.41 秒。这个结果是用于暴露模型和数据边界的独立检查，不是把未知值当成正确答案。

## 结果文件

| 文件 | 内容 |
|---|---|
| [报告 PDF](outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Vision_II_Review_2026-10-22.pdf) / [Word](outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Vision_II_Review_2026-10-22.docx) | 10 月 22 日更新的 12 页审阅稿，已纳入单评审诊断及落地改进；待组员审核 |
| [提交准备与未完成项](docs/SUBMISSION_READINESS.md) / [视频展示提纲](docs/VIDEO_PRESENTATION_OUTLINE_2026-10-21.md) | 报告、视频、代码和数据的具体状态及咨询待确认要求 |
| [报告初稿（2026-10-29）](docs/REPORT_DRAFT_2026-10-29.md) | 按老师六章节重新整理，纳入最终训练审计、严格 unknown 和独立测试边界；待组员补成员贡献、引用和最终表述 |
| [PPT 初稿源稿](docs/PRESENTATION_DRAFT_2026-10-29.md) / [可编辑 PPTX](docs/PRESENTATION_DRAFT_2026-10-29.pptx) | 12 页、十分钟版本；使用真实 EDA 图表，替换成员姓名后排练 |
| [最终视频大纲](docs/VIDEO_OUTLINE_2026-10-29.md) | 十分钟真人讲述时间轴、画面和讲述要点；现有无声底片不是最终视频 |
| [最终增广训练审计](docs/FINAL_AUGMENTED_TRAINING_2026-10-29.md) | 球和门将两组增广对照、选择规则、未晋级原因及 label-free smoke 结果 |
| [provisional coverage 说明](outputs/provisional_coverage_2026-10-29/README.md) | 17/39 高度候选和 27/39 最后可见球帧仅作为 provisional observation，不是准确率 |
| [GitHub 发布范围](docs/GITHUB_PUBLISH_SCOPE_2026-10-29.md) | 说明本次保留的代码、审计和初稿，以及原视频/权重/压缩包的排除原因 |
| [人工标注整合及新诊断](docs/HUMAN_REFERENCE_INTEGRATION_2026-10-22.md) / [保存结果审阅 ZIP](outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Saved_Result_Review_2026-10-22.zip) | 标注来源、单评审对照、复现包与边界；未上传 |
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

上述历史主结果使用已有权重推理，当时没有训练或微调模型。10 月 26 日的新增监督模型单独记录。比分板处理使用图标几何和颜色。SAM2 的轮廓试验单独保存，完整扑救高度尚未测出。具体参数见[主实验设置](docs/FRESH_EXPERIMENT_PROTOCOL.md)和[连续助跑实验](docs/SHOE_APPROACH_RESULTS_2026-10-13.md)。

## 当前指标

以下数量表示有候选输出的罚球数。覆盖率以 39 次罚球为分母。10 月 22 日已整合一名人工评审的 39 段标注；双评审共识和独立测试准确率仍未建立。与单评审对照，脚别补充 12/14、门将方向 19/23 一致。27 个可比较窗口中，22 个落地候选匹配人工列表 156 次中的 13 次，列表计数 MAE 4.963；落地仍明显漏检。[完整诊断](docs/HUMAN_REFERENCE_INTEGRATION_2026-10-22.md)保留未知、分歧及人工/AI 辅助来源。

随后完成的[固定脚跟/脚尖支撑补充](docs/DEVELOPMENT_COMPLETION_2026-10-22.md)保留原候选并增加 43 个，总计 65 个、覆盖 23 段；同 27 窗口匹配数提升到 41，列表计数 MAE 降至 3.370。下表继续描述冻结主结果，新增开发实验单独记录，不混写版本。

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

赛果一致率是在参与开发的这批视频上计算的，尚无独立测试集。比分板和融合赛果各覆盖 35 段，两组事件有所不同：Koke 的球轨迹与比分板冲突，融合结果保留未知；Rakitic 的比分板缺失，由球轨迹规则补入。数据见[主结果表](outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.csv)；最新[人工评估状态](outputs/fresh_course_experiment/human_evaluation_2026-10-22/evaluation_consensus.json)为 39 段单评审已完成，双评审共识为空。旧 `evaluation_pending.json` 仅为保留的历史记录。

## 主要问题与后续工作

旧程序满足三帧低运动条件后就截断支撑记录，导致正常与镜像两路起点相差一帧时漏配。10 月 18 日的修复保留同一阶段的完整支撑，并检查共同持续时间。原有 12 个候选全部保留，增加 10 个。[回放对照](docs/SHOE_LANDING_REPLAY_2026-10-18.md)记录了每段变化。

Rashford 的旧脚踝计数为 15，鞋部候选由 3 个变为 6 个。新增候选中仍有仅约 0.067 秒的短支撑，空中变向也可能满足低运动条件。新人工列表已用于新版鞋部候选诊断，仍须逐项分析漏检和错误支撑；主表的旧计数保留原值。

脚别覆盖从 8 段增加到 13 段，再增加到 14 段，剩余 25 段未知。新增的 Dzagoev 右脚候选依赖第 262 帧的近球证据；Kane 仍然未知。后续先核对原录像里的触球区间、下肢遮挡和人物身份，再针对失败片段调整裁剪与跟踪。

地面投影和门将高度都依赖相机几何。目前地面标定在 609 个采样帧中接受了 521 帧，仍需核对投影误差与时间连续性。门将高度缺少完整落地阶段和稳定地面参考。下一步按完整动作重新检查镜头和人物连续性，跨镜头结果分别保存。

已有一名评审提供原课程视频的落地帧、触球脚和四项定性射门/扑救标签。新[整合脚本](scripts/integrate_human_reference_2026_10_22.py)检查原文件哈希、来源和坐标语义，并分别计算单评审脚别、门将方向及落地列表诊断。原[评估脚本](scripts/evaluate_shoe_landings.py)的双评审共识规则保持不变；第二人复核后可按其格式保留分歧并重新评价。不要拿历史空模板覆盖已经完成的人工标签。

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
