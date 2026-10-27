# 10 月 22 日人工标注整合与诊断

已发现并整合另外提供的 `VisionII_39_人工标注_unknown修订版_20261021`。旧报告的“人工标注为 0”已过时，不能继续作为当前状态。

## 已完成

- 47 个源文件哈希全部核对，39 个视频哈希与冻结预测完全对应；原始包逐字节保留。
- 39 段均有一名评审的脚别及落地标注；223 个窗口内落地和 5 个窗口外观察，共 228 个。
- 射门高度、扑救高度、球水平方向、门将方向共 156 项已标注。Aspas 和 Fernandes 的出界射门按原作者规则保留 `unknown`，这不是漏标。
- 用原评价器重算：39 段已复核、0 段待标。双评审共识指标仍为空，因为只有一名评审。
- 新增单评审诊断、逐事件 CSV 和两张对照图，更新英文 PDF/Word。没有重跑神经网络，也没有用人工标签覆盖自动输出。

## 实际诊断

| 比较 | 结果 | 解释 |
| --- | --- | --- |
| 最新脚别候选 | 14/39 有输出，12/14 与人工相同 | Golovin、Brozovic 不同，保留原预测与分歧 |
| 门将画面方向 | 23/39 有输出，19/23 相同 | 统一使用画面左/右；有 4 段分歧 |
| 落地列表 | 27 个可比较窗口；22 候选，对应人工列表 156 次 | 0.08 秒一对一匹配：13 匹配、9 未匹配预测、143 未匹配人工项 |
| 落地列表诊断 | 精确率 59.1%、召回率 8.3%、计数 MAE 4.963 | 仅单评审提供列表上的开发集诊断；12 段预测不可用不填零 |

单评审曾看过汇总候选数，不是全盲评估。大多数完整助跑可见性由 AI 辅助判定，不能写成第二名人工复核或完整助跑准确率。定性高/中/低没有米制标定和精确证据帧，不能把人工表当成自动完成高度估计。

## 可打开的文件

- [新版报告 PDF](../outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Vision_II_Review_2026-10-22.pdf)
- [新版可编辑 Word](../outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Vision_II_Review_2026-10-22.docx)
- [整合与诊断 JSON](../outputs/fresh_course_experiment/human_evaluation_2026-10-22/integration_summary.json)
- [39 段逐事件对照 CSV](../outputs/fresh_course_experiment/human_evaluation_2026-10-22/per_event_comparison.csv)
- [未改动的人工标注副本](../outputs/fresh_course_experiment/human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json)
- [原评价器输出](../outputs/fresh_course_experiment/human_evaluation_2026-10-22/evaluation_consensus.json)

当前报告为 12 页。Canvas 的原文是一般指引不超过 20–25 页，不是至少 20 页；本次只增加实际证据、方法和结果。之后补做的[固定点落地改进](DEVELOPMENT_COMPLETION_2026-10-22.md)将列表计数 MAE 从 4.963 降到 3.370，旧基线仍保留。

仍需完成：落地算法的漏检分析与改进、完整俯视轨迹/严格高度/最终球位置/慢动作要求的处理、完整原片及全部提取流程复现、本人或组员真人讲述、真实成员与贡献及最终审核。人工标注本身已经完成，不应再要求把同一批 156 项从头重标。
