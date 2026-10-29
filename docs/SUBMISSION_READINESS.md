# IND5003 Vision II 提交准备

核对日期：2026-10-22，新加坡时间。Canvas 的 Project Report 仍未提交，截止为 **10 月 30 日 23:59**。本轮已整合另外提供的 39 段人工标注、计算单评审诊断并更新报告；整项小组项目尚未做完。

## 现在可以直接看的文件

| 文件 | 当前状态 |
| --- | --- |
| [更新后的英文 PDF](../outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Vision_II_Review_2026-10-22.pdf) / [Word](../outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Vision_II_Review_2026-10-22.docx) | 12 页、六个规定章节、Times New Roman 11pt；已纳入人工对照、落地改进及具体缺口，仍需组员审核和成员信息 |
| [报告源稿](REPORT_FRESH_REVIEW_DRAFT.md) | 与新 PDF/Word 同一版，包含本轮 AI 使用声明 |
| [39 次罚球主结果](../outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.csv) | 已有自动候选及来源记录；不是全部特征都已完成 |
| [最新脚别补充](../outputs/fresh_course_experiment/shoe_approach_foot_supplement_v2/foot_candidates_39.csv) | 14/39 候选，10 右、4 左；其余 25 未知 |
| [10/18 落地修复对照](../outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1/comparison_39.csv) | 同一输入下 12 个候选增至 22 个，覆盖事件 7 增至 12；没有重新运行神经网络 |
| [原结果查看页](../outputs/fresh_course_experiment/review_v3_shoe_fusion_ui3/index.html) | 可看既有图片和记录；视频需另备原课程片段 |
| [10 分钟视频提纲](VIDEO_PRESENTATION_OUTLINE_2026-10-21.md) | 新补的分镜和讲述要点；不是已经录制的最终视频 |
| [4 分 30 秒无声审阅视频](../outputs/fresh_course_experiment/submission_review_2026-10-21/IND5003_Vision_II_Silent_Review_2026-10-21.mp4) | 真实原片正常速度、分析图与边界说明；需真人讲述和最终剪辑，不能当作已完成的视频作业 |
| [本轮核验记录](../outputs/fresh_course_experiment/submission_review_2026-10-22/verification.json) | 主表 4 个文件哈希、39 个回放预测、39 个脚别补充均通过核对；人工标注已整合为 39 段 |
| [人工整合与单评审诊断](HUMAN_REFERENCE_INTEGRATION_2026-10-22.md) / [逐事件 CSV](../outputs/fresh_course_experiment/human_evaluation_2026-10-22/per_event_comparison.csv) | 47 个源文件哈希与 39 个视频来源对应通过；脚别 12/14、门将方向 19/23 与人工相同；落地仍有严重漏检 |
| [保存结果审阅 ZIP](../outputs/fresh_course_experiment/submission_review_2026-10-22/IND5003_Saved_Result_Review_2026-10-22.zip) / [包核验回执](../outputs/fresh_course_experiment/submission_review_2026-10-22/package_verification.json) | 源码、配置、结果、人工标签、报告、底片和两段本地视频；可重现保存结果分析，不含完整原片和模型 |

正式 [project brief](ind5003_vision_ii_project_requirements.pdf) 第 5–6 页规定报告**最多 25 页**，封面、目录、参考文献和附录不计；最新 Canvas 的一般指引同样是不超过 20–25 页，并未规定至少 20 页。12 页稿可以作为当前审阅基础，但不能靠篇幅判断内容已满足所有要求。

## 本轮已补齐

1. 整合原先位于 Downloads 的 v040 人工包，47 个来源文件均按哈希核对，原始文件未修改。39 段、156 项定性字段、228 次落地观察已经完成，不应再记为 0。
2. 计算独立命名的单评审诊断：脚别 12/14 与人工一致；门将方向 19/23；27 个可比较窗口中落地候选只匹配人工列表的 13/156 次，列表计数 MAE 4.963。12 段无预测保持未知，未拿人工标签替换自动输出。
3. 继续实现固定脚跟/脚尖支撑补充，在相同姿态和阈值下把匹配数 13 提升至 41，列表计数 MAE 从 4.963 降至 3.370。旧预测保留，43 个新增候选单独记录；仍未解决全部落地。[方法、前后对照与具体素材缺口](DEVELOPMENT_COMPLETION_2026-10-22.md)。
4. 更新报告为 12 页，加入三张人工/开发对照图及 AI 使用披露。已有真人录制分镜和 4 分 30 秒无声底片继续可用。
5. 针对标注、姿态支撑、旧回放、结果完整性的测试及解压重现均记录在包核验回执；保存结果包可重跑 EDA、人工诊断、新姿态回放和具体缺口审计。这些验证不代表全原片模型推理已复现。

## 还没有完成的部分

| 要求 | 实际现状 | 完成所需证据 |
| --- | --- | --- |
| 单评审及独立复核 | 单评审 39 段已完成，228 次观察及 156 个定性字段均保留；原模板中的 0 是历史快照 | 可进行第二人复核以量化分歧；不能把 AI 辅助可见性判断当作第二人，也不应要求同一批标签从头重做 |
| 完整俯视助跑轨迹 | 32/39 有触球前地面候选点；连续轨迹和投影准确率未验证 | 逐镜头校准、误差检查、连续地面轨迹及失败说明 |
| 门将完整扑救高度 | strict height 0/39；仅 1 段局部且带条件的图像信号 | 先确认高度定义和可见完整动作，再提取、校准和评估；不可把像素 proxy 当米 |
| 球事件最终位置 | verified final position 0/39 | 检查 play-end 边界、遮挡与切镜；最后一次检测不能自动等于最终位置 |
| 四类镜头时间及慢动作 | 有空间视角候选；320 个区间的自动慢动作判断全部未知 | 跨镜头动作对齐或可验证的播放速度判据，并检查时间区间 |
| 最终展示视频 | 已有本轮无声审阅底片，未有最终真人讲述版 | 小组真人录制，最多 10 分钟，不加速；原题第 6 页明确“avoid using an AI voice”，不强制露脸或人人讲 |
| 正式代码与数据包 | 已生成并验证保存结果审阅包；本地只有 Rashford 与 Muriel 两段片段，哈希一致；完整数据及原片到特征的复现未完成 | 补授权课程完整原片及模型依赖，在另一目录复现原片到特征完整路径 |
| 封面、个人贡献及最终声明 | 不推定组员身份和贡献 | 小组提供真实成员信息，确认实际分工，审核 AI 使用披露 |

其余自动候选覆盖为：触球 33/39、旧版助跑计数 32/39、门将触球前移动 29/39、画面扑救方向 23/39、球轨迹 33/39、融合赛果 35/39。35/35 的外部赛果一致是开发集描述，不是独立测试准确率。未知保持未知，不能填零或人工补表后冒充自动提取。

## 咨询后待确认的要求

此前 Group 19 安排 **10 月 22 日 20:00–20:15** 咨询。本地没有该场咨询的实际结论记录，以下问题仍需按老师真实答复处理：

1. 广播片段无法观测完整扑救高度或事件终点时，是否接受可观察定义、弃权和失败分析；若不接受，应优先补哪类证据？
2. 俯视助跑与门将高度期望使用什么物理单位，是否要求完整连续轨迹或可接受分段输出？
3. 四类镜头中慢动作如何界定，镜头的原始事件匹配是否可以人工核验，而最终分类仍由脚本完成？
4. 当前以可观测性、计数定义敏感性、赛果证据融合为三个问题，是否符合 Vision II 的预期分析深度？
5. 最终 SharePoint 的代码、原视频、模型文件应该怎样组织，个人 quiz 与 TEAMMATES survey 的开放时间在哪里确认？

以上为待询问事项，不能当作教师已允许豁免要求。

## 实际提交前的动作

当前真实标注已保存在[整合后的 v040 原文件](../outputs/fresh_course_experiment/human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json)，不要用历史空模板覆盖它。新增评审时按[标注格式说明](SHOE_LANDING_REFERENCE_SCHEMA.md)保留不同评审和分歧，再运行评价；保持开发集评价边界。

报告上传 Canvas；视频、源码和数据按 brief 上传 SharePoint。三项均为 10 月 30 日 23:59 截止。随后还有个人 quiz（10%）和 TEAMMATES 同伴评价（5%），具体开放及截止时间以课程通知为准。文件生成不等于提交，本轮没有执行上传。
