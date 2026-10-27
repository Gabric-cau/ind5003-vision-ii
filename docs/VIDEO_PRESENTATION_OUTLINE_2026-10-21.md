# Vision II: 9 分 30 秒真人展示提纲

状态：可录制提纲，尚未生成真人讲解视频。按真实理解自然讲述；若出现人脸画面，不要照稿朗读。保留正常播放速度，不使用 AI 配音。画面只展示实际已有输出，把候选、未知和未验证准确率讲清楚。原题第 6 页说明不要求人人讲述，也不强制露脸。

已制作 [4 分 30 秒无声审阅底片](../outputs/fresh_course_experiment/submission_review_2026-10-21/IND5003_Vision_II_Silent_Review_2026-10-21.mp4)，包含 Rashford 32 秒、Muriel 34 秒正常速度原片，以及真实结果图、失败例和未完成状态。原片哈希与冻结实验一致；详细片段时长和来源见 [视频清单](../outputs/fresh_course_experiment/submission_review_2026-10-21/silent_video_manifest.json)。它是素材审阅底片，可在最终剪辑中调整静态图停留时间并加入真人讲述、实际代码操作；不是十分钟正式视频的替代品。

| 时间 | 屏幕内容 | 自然讲述要点 |
| --- | --- | --- |
| 0:00–0:40 | [覆盖图](../outputs/fresh_course_experiment/final_eda_v3_shoe_fusion/01_fresh_coverage.png) | 四场比赛、39 次罚球。主要发现是特征可见性很不均匀：触球 33/39，融合赛果 35/39，完整扑救高度和事件最终球位置仍无验证输出。 |
| 0:40–1:30 | 一段课程原片和对应 clip ID | 明确原片、片段边界、帧号和 30 FPS 时钟。介绍允许人工切片，后续特征由脚本产生；同一罚球的不同视角不算独立样本。 |
| 1:30–2:30 | [结果查看页](../outputs/fresh_course_experiment/review_v3_shoe_fusion_ui3/index.html)，选一个有触球的事件 | 人物检测、身份连续性、静止球到离开的时间窗口、门将和球场几何分别做什么。让观众看到真实输入和对应候选，避免仅展示最终表。 |
| 2:30–3:45 | [新旧落地对照图](../outputs/fresh_course_experiment/submission_review_2026-10-21/landing_replay_comparison.png)，Rashford 原画面 | 旧程序三帧就截断，镜像两路相差一帧会丢失共同支撑；完整记录修复后 12→22 个候选。Rashford 的旧脚踝极值为 15、原鞋部候选为 3、新候选为 6；这些都不是已核验真实步数。 |
| 3:45–5:00 | [赛果矩阵](../outputs/fresh_course_experiment/final_eda_v3_shoe_fusion/05_fresh_scoreboard_outcomes.png)，Koke 一行 | 球轨迹规则覆盖10次、与参考一致9次；比分板覆盖35次。组合对 Koke 冲突弃权，由球规则补入克罗地亚对丹麦的 Rakitic，因此覆盖同为35但集合不同。35/35 是参与开发视频上的参考一致，不是测试准确率。 |
| 5:00–6:05 | [步数与时间图](../outputs/fresh_course_experiment/final_eda_v3_shoe_fusion/02_fresh_steps_timing.png) | 三个问题：哪些特征可观察？触球与动作阶段定义如何影响计数？不同来源证据能否减少赛果弃权？解释 EDA 指向的变化，不从候选计数推导最佳罚球策略。 |
| 6:05–7:10 | 一个高度或最终位置失败例 | 指出遮挡、镜头切换、未看到动作终点。稀疏地面点不是连续俯视路径，最后观测球点不是事件最终位置，局部像素高度不是完整扑救高度。 |
| 7:10–8:20 | 终端运行下方保存结果分析命令，展示生成图和核验记录 | 清楚说这是从保存的结果生成图，并非重新推理原片。介绍 `src`、`scripts`、`configs`、依赖文件、哈希和测试各自用途。完整原片复现需要另外交付数据和权重。 |
| 8:20–9:10 | [标注模板](../outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/human_reference_template.json)，隐藏预测的复核页 | 真人独立标注仍未完成；评价保留分歧，未知不填零。说明将如何算落地精确率、召回率、时间误差及可见区间步数 MAE。 |
| 9:10–9:30 | 结尾图：已有成果与未完成测量 | 用真实状态收尾：已有可追溯候选与证据，部分测量仍需改进。按实际情况说明 AI 辅助、组员贡献与后续工作。 |

在项目根目录中运行保存结果的展示命令：

```sh
python -m src.fresh_eda \
  --bundle-dir outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion \
  --output-dir outputs/video_demo_eda
```

先在本机完成一次实际演练，再录制。命令需要 `requirements-analysis.txt`，不会重新分析原视频。使用新输出目录可以保留既有正式图表。

录制前需要小组补齐：真实成员/分工、可播放的授权原片、对每张图的理解、最终采用的分析版本。若后续实验或人工评价完成，应同步替换图、数字和讲述，不能保留本提纲中的旧待办状态。
