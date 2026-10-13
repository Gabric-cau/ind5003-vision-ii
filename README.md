# IND5003 Vision II：点球视频分析

这是 IND5003 Vision II 的点球视频分析项目，使用 2018 年世界杯四场点球大战的 39 次罚球：哥伦比亚对英格兰、克罗地亚对丹麦、西班牙对俄罗斯，以及俄罗斯对克罗地亚。

一脚点球从助跑到扑救只有几秒，但里面有不少值得比较的细节：球员走了多少步、用哪只脚触球、门将什么时候开始移动。项目尝试把这些细节从转播画面里提取出来，再看看它们和射门结果有没有关系。代码在小组原项目的基础上继续修改，当前版本重新处理了全部 39 次罚球。

## 查看结果

| 文件 | 内容 |
|---|---|
| [报告 PDF](outputs/fresh_course_experiment/report_v4_continuous_shoe/REPORT_FRESH_REVIEW_DRAFT.pdf) / [Word](outputs/fresh_course_experiment/report_v4_continuous_shoe/REPORT_FRESH_REVIEW_DRAFT.docx) | 方法、分析和遇到的问题，目前为审阅稿 |
| [结果表](outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.csv) | 每次罚球一行，共 39 行 |
| [完整 JSON](outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.json) | 各模块的输出、帧号及来源记录 |
| [逐段查看](outputs/fresh_course_experiment/review_v3_shoe_fusion_ui3/index.html) | 下载仓库后在浏览器打开，查看检测图片和对应结果 |
| [统计图](outputs/fresh_course_experiment/final_eda_v3_shoe_fusion/) | 特征覆盖、步数比较和赛果分析 |
| [连续助跑实验](docs/SHOE_APPROACH_RESULTS_2026-10-13.md) / [原画面对照](outputs/fresh_course_experiment/shoe_approach_analysis_v2/review_v2/index.html) | 原课程视频上的新推理，逐段比较脚踝极值与落地候选 |
| [人工复核](outputs/fresh_course_experiment/shoe_approach_analysis_v2/blind_review/index.html) | 本地原视频，不显示预测；尚未完成人工标注 |
| [脚别补充表](outputs/fresh_course_experiment/shoe_approach_foot_supplement_v2/foot_candidates_39.csv) | 保留原 13 个候选，增加 Dzagoev，共 14/39；未经人工确认 |

下面是 Kane 罚球片段的检测输出，图中保留了视频帧号，方便回看原画面。

![Kane 罚球片段的检测输出](outputs/fresh_course_experiment/review_v1/evidence/col_eng_p02_kane_eng9_evidence.jpg)

## 怎么看当前结果

视频按罚球切片后，用 YOLO11 提取人物姿态和足球位置。脚本根据球从静止到离开罚球点的过程估计触球时刻，再计算助跑步数和门将动作。现在有 33 次罚球给出了触球时间候选，32 次给出了步数候选；门将触球前移动和扑救方向分别有 29 次、23 次输出。具体数字都在结果表里，遮挡和判断冲突的片段保留为 `unknown`。

现在看，结果还很不均衡。左右脚这一列尤其需要核对：原来的脚踝规则得到 8 次候选，加入脚跟和脚尖后补上了 5 次，当前共 13 次。离球最近的脚可能是支撑脚，射门后的摆腿和支撑脚回收也会混在同一个运动窗口里。Rashford 在新方法下有了右脚候选，Kane 仍然未知。人物检测得清楚不等于触球脚就能判断正确，触球前后的动作还是要分开看。

步数里，Rashford 的 15 次和 Pivaric 的 13 次需要重点检查。它们在三个阈值设置下都得到相同数字，但算法目前数的是脚踝信号的极值，还不能确认每个极值都对应一次落地。可能有连续的小碎步，也可能把一次摆动重复计数。看到这种数字，还是得回到原画面逐帧检查，不能只因为几次运行都一样就认为数对了。

按赛果分组，有步数输出的 20 次进球中位数是 5，12 次未进球中位数是 5.5。这个差别不大，而且两组缺失的片段不一样，步数本身也还没核实。现阶段据此讨论哪种助跑更容易进球，结论会太早。[步数比较图](outputs/fresh_course_experiment/final_eda_v3_shoe_fusion/04_fresh_approach_outcomes.png)保留了这部分观察。

进球这一列看起来最完整：从转播比分板的红绿图标读出了 35 次结果，与 FIFA 技术报告的记录一致。不过球轨迹规则本身只对 10 次罚球作出了判断。Koke 这一球尤其说明问题：球轨迹给出了进球候选，比分板显示未进，两者冲突，合并结果就保留了未知。比分板能补赛果，却不能证明球已经跟踪正确。35/35 是这批调试视频上的一致性，还没有在新比赛上验证。

俯视图目前只有 32 次罚球的部分地面投影点，不能连成完整的助跑路线。高度检查发现，原来 6 段局部信号里有 5 段的地面参考已经失效，旧代码却仍给出了“低扑救”。现在已撤回这 5 个类别，只保留 Brozovic 一段条件信号，原始数值留作诊断。完整扑救、慢动作和球的最终位置仍没有可用的自动结果。门将方向按画面左、右记录，步数只统计可见的助跑部分。

## 10 月 13 日的新实验

左右脚先试了一项很小但有针对性的改变：投票在球第一次离开时停止，不再混入后续支撑腿回弹。Trippier、Dier 和西班牙对俄罗斯的 Smolov 得到了新的右脚候选，但窄窗口也失去了旧规则的八个候选。这个结果提醒我，缩小窗口能解释一些失败，却不是直接提升整体识别率。三段的[逐帧证据和完整比较](outputs/fresh_course_experiment/foot_strike_window_verified_inputs_v1/README.md)单独保留，主表还没有采用它们。

接着用 MediaPipe 的 33 个关键点重新处理了全部 39 段，增加了脚跟和脚尖的位置。两路独立的正常、镜像推理使用固定裁剪，并检查人物身份和原画面里的球。Rashford、Dier、Pivaric、西班牙对俄罗斯的 Smolov 和 Vida 同时通过摆腿与鞋区域检查，补到了主表；新增的是四个右脚、一个左脚候选。Ramos 只有摆腿证据，没有通过鞋区域检查，所以没有靠新模型补值。这次覆盖从 8/39 提高到 13/39，但还有 26 段未知，也还没有独立人工标签来算准确率。[实验设置](docs/FRESH_SHOE_POSE33_PROTOCOL_2026-10-13.md)和[五段原画面对照](outputs/fresh_course_experiment/shoe33_guarded_source_review_v1/README.md)保留了判断依据。

随后把推理窗口向前延伸到可见助跑，仍然只用原来的四份课程录像。32 段有可用区间，共重新处理 2,121 帧，完成 4,242 次正常、镜像推理。落地规则改为寻找鞋从摆动转为短暂低运动的过程，并检查相机运动和重复画面；7 段留下了 12 个候选。Rashford 和 Pivaric 各有 3 个，与旧规则的 15、13 相差很大。但这些支撑都只有三帧，约 0.067 秒，脚在空中改变方向也可能出现低运动，所以现在不能把 3 写成真实步数。另有 20 段没有检测到符合条件的候选，这也不代表没有迈步。

连续推理还在 Dzagoev 一段得到新的右脚候选，近球证据在第 262 帧，位于球的静止帧 261 与离开帧 263 之间。它说明较长的追踪历史会改变模型输出，但还不能证明脚和球实际接触。新实验保留了逐帧来源、正常与镜像坐标以及拒绝原因；[复核模板和评估结果](outputs/fresh_course_experiment/shoe_approach_analysis_v2/evaluation_pending.json)目前 39 段均待人工复核，准确率和步数误差留空。

重新解码原视频、核对模型坐标和旧输出后，脚别补充表保留原来的 13 个候选，只新增 Dzagoev，合计 14/39，没有已确定脚别之间的冲突。主表仍保留原版本，便于检查到底增加了什么。候选覆盖只多了一段，离可靠地分析全部罚球还有明显距离。

门将局部推理在 Kane、Rashford 和 Cuadrado 三段上加入了旋转视角。Kane 的触球前身体参考终于能建立，但落地阶段仍然不完整，三段都没有新增高度类别。人物身份检查也收紧了：新检测只能改善同一帧的原人物，不能给裁判换个 ID 后把他当门将。这轮结果比增加一个看似完整的数字更有用，它把剩下的问题定位到了动作连续性和地面参考。

高清录像也实际跑了新推理。前三球中，中型足球检测器给出了 Falcao、Kane 两段触球候选，小型检测器只给出 Falcao，Cuadrado 两者都没有解决。Kane 的高清主视角仍没有得到可靠脚别。换高清素材有帮助，但清晰度本身解决不了遮挡和关键点跳变，[三球对照](docs/HD_PILOT_RESULTS_2026-10-13.md)记录了这些没有成功的尝试。

## 接下来怎么核对

这轮实验没有新增视频。接下来先由两名同学分别查看原课程片段，记录能看清的落地帧和触球脚，再比较自动结果。复核时先隐藏预测和叠图，避免照着模型填答案。已经看过这些视频、调整过方法，所以即使人工复核后表现不错，也只能报告这批视频上的一致性。

早先的 FIFA 高清试验单独保留，没有进入主表或本轮连续助跑实验。课程简章允许补充数据作为证据和背景，但具体提交范围仍应遵守教学团队的要求。[素材说明](docs/ADDITIONAL_VIDEO_SOURCES.md)区分了原课程录像和外部剪辑；不能把不同剪辑的时间、位置直接混用，也不能用额外比赛代替指定的 39 次罚球。

## 在本机运行

查看报告、结果表和检测图片不需要 GPU。重新生成统计图可以直接使用仓库里的结果文件：

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

Windows 的虚拟环境激活命令为 `.venv-analysis\Scripts\Activate.ps1`。统计图会写入 `outputs/my_eda`，重复运行会覆盖同名文件。

如果要运行测试，再安装 OpenCV 和 SciPy：

```sh
python -m pip install -r requirements-experiments.txt
python -m pytest -q
```

从原视频重新提取特征需要另外准备视频、模型权重和 FFmpeg。各阶段的脚本位于 `scripts/run_fresh_*`，运行参数可通过 `--help` 查看。具体的特征定义和实验设置见[实验说明](docs/FRESH_EXPERIMENT_PROTOCOL.md)。

## 文件组织与来源

`src/` 放分析模块，`scripts/` 放运行脚本，`configs/` 放切片和分析配置，`tests/` 放测试。当前结果统一放在 `outputs/fresh_course_experiment/` 下，其他实验目录保留用于比较早期版本。

原小组仓库：[RX422040/vision-ii-penalty-analysis](https://github.com/RX422040/vision-ii-penalty-analysis)。这里保留了原提交历史，后续修改主要涉及触球时间、足球跟踪、相机运动补偿、地面标定、比分板分析和结果整理。代码与文档更新使用了 AI 辅助。
