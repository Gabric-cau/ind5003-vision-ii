# 连续助跑落地候选：完整支撑回放

本次修复落地候选的证据收集和配对逻辑，重新处理原课程 39 段视频保存的关键点。没有增加视频、重新运行神经网络或修改主结果表。脚别仍使用原有输出。

## 修复内容

旧程序一满足三帧低运动条件就结束该候选的支撑记录。正常和镜像两路即使识别到同一段持续站立，起点相差一帧时，两个三帧集合也只共同覆盖两帧，会被配对规则拒绝。

新版保留首次候选时刻，继续记录同一次低运动阶段的后续观测。新的摆动、身份缺失或拒绝、相机断链以及超出散布范围都会结束该段支撑。配对先检查共同帧数、共同持续时间和连续性，再选择时间最近的候选。判定阈值与旧实验相同。

例如 Rashford 的右脚两路分别从第 229、228 帧开始记录。完整支撑记录允许检查第 229–231 帧的共同证据，旧程序截断后的交集不足三帧。

## 回放结果

| 项目 | 旧版 | 修复后 |
|---|---:|---:|
| 课程罚球总数 | 39 | 39 |
| 有连续推理输入的片段 | 32 | 32 |
| 有配对落地候选的片段 | 7 | 12 |
| 配对候选总数 | 12 | 22 |
| 可处理但未检出候选的片段 | 20 | 15 |
| 左右通道时序歧义 | 5 | 5 |
| 无可用输入区间 | 7 | 7 |

原有 12 个候选的帧号和脚别全部保留。新增候选分布如下：

| 片段 | 候选数变化 | 新增候选的共同支撑时间 |
|---|---:|---|
| Falcao，哥伦比亚对英格兰 | 0 → 1 | 0.067 秒 |
| Rashford | 3 → 6 | 0.067、0.100、0.067 秒 |
| Smolov，西班牙对俄罗斯 | 1 → 3 | 0.233、0.200 秒 |
| Koke | 0 → 1 | 0.067 秒 |
| Golovin | 0 → 1 | 0.067 秒 |
| Fernandes | 0 → 1 | 0.067 秒 |
| Modric，俄罗斯对克罗地亚 | 0 → 1 | 0.133 秒 |

旧版源代码快照也重新执行了全部 32 个可用窗口，状态、候选数量和完整候选内容均与冻结结果一致。新旧比较使用相同的姿态、相机变换、人物身份约束、解码画面哈希和助跑区间。

## 结果边界

这次增加来自配对逻辑修复。图像中的短暂低运动仍可能出现在空中变向或关键点抖动中，候选数增加不能证明真实步数更准确。Rashford 的旧脚踝极值计数为 15，修复后的鞋部候选为 6，两者仍需逐帧标注后比较。

独立人工标注仍为 0 段。准确率、落地事件精确率和召回率、步数平均绝对误差保持空值。五段左右通道歧义和七段不可用区间保留未知，未转成零步。该批视频已经参与开发，回放不属于独立测试。

本轮还检查了相机缓存：主结果 39 段的相机链均完整；连续助跑 32 段共有 6,503 对相邻帧，没有被拒绝的帧对或空变换。相机重定位没有在本批数据上显示出可恢复的断链，因此本次没有修改相机算法。

## 文件与复现

- [39 段新旧对照表](../outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/comparison_39.csv)
- [汇总与输入记录](../outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/summary.json)
- [本版预测冻结记录](../outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/prediction_freeze.json) / [待填写标注模板](../outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/human_reference_template.json)
- [待标注评估结果](../outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/evaluation_pending.json)
- [Rashford 原画面与叠图](../outputs/fresh_course_experiment/shoe_landing_source_review_2026-10-18_v1/col_eng_p04_rashford_eng19.jpg)

回放只需已有的关键点和结果文件：

```sh
python -m scripts.replay_shoe_landings \
  --dense-dir outputs/fresh_course_experiment/shoe_approach33_cohort_v1 \
  --baseline-analysis-dir outputs/fresh_course_experiment/shoe_approach_analysis_v2 \
  --output-dir outputs/my_landing_replay
```

输出目录必须尚不存在。原视频查看可使用[原画面复核页](../outputs/fresh_course_experiment/shoe_approach_analysis_v2/blind_review/index.html)，原视频需另行准备。评估本轮结果需填写本轮模板，旧版模板绑定的是修复前的预测。模板中的评审身份、时间和判断留待真人填写。
