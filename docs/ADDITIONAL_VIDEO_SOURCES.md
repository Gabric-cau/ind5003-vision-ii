# 补充视频与后续实验

这份清单记录目前能找到的素材，以及它们可能帮助解决的问题。10 月 13 日已下载哥伦比亚对英格兰的 FIFA 官方录像，并切出前三球开展独立推理。其他链接仍只是访问与元数据检查。新增录像尚未加入课程的 39 行主结果表。

## 课程允许补充数据吗

[课程说明](ind5003_vision_ii_project_requirements.pdf)第 6 页的 Data overview 鼓励寻找相关数据，为结论提供额外证据或背景，并说明这会获得加分。第 4、5 页仍将 Vision II 的材料指定为课程提供的四场 2018 世界杯点球大战。文档没有明确允许用其他比赛替换它们。

因此，原来的四场 39 次罚球继续作为主数据。同场完整录像用于补充观察；其他比赛或带标注的数据用于测试方法。多个机位、同一球的回放、不同分辨率副本，都不会增加独立罚球的样本数。

## 同场官方录像

| 比赛 | 素材 | 已确认的内容 |
|---|---|---|
| 哥伦比亚对英格兰 | [FIFA 完整点球大战](https://www.youtube.com/watch?v=NtvUzy00DuU) | 已下载公开的 1920×1080、50 fps 版本，并顺序解码 25,607 帧 |
| 克罗地亚对丹麦 | [FIFA 完整点球大战](https://www.youtube.com/watch?v=lO36Q8Uj2bE) | FIFA 频道的公开媒体元数据可访问，最高标准 H.264 1080p25；尚未下载 |
| 克罗地亚对丹麦 | [FIFA+ 全场回放](https://www.plus.fifa.com/en/content/croatia-v-denmark-round-of-16-2018-fifa-world-cup-russia-full-match-replay/718d3bd3-8cf2-4639-b2c6-af6d5ccaab48) | 官方页面被索引，本次直接访问受限 |
| 西班牙对俄罗斯 | [FIFA+ 全场回放](https://www.plus.fifa.com/en/content/spain-v-russia-round-of-16-2018-fifa-world-cup-russia-full-match-replay/b6b1bccf-72fd-4f60-a112-9242c6cd21c0) | 官方页面被索引，本次直接访问受限 |
| 俄罗斯对克罗地亚 | [FIFA+ 全场回放](https://www.plus.fifa.com/en/content/russia-v-croatia-quarter-finals-2018-fifa-world-cup-russiatm-full-match-replay/49537884-4db8-4588-9243-c53a5a8b7f63) | 官方页面被索引，本次直接访问受限 |

本地四个主视频实际均为 1280×720、30 fps。新下载文件为 268,730,912 bytes，SHA256 为 `6075b0d85ce05bc01880a395fea9e61e502ad20ab766cb4c34761cfa77f4b0f3`。完整顺序解码未发现相邻帧完全重复，全片运动强度也没有简单的奇偶交替重复规律。这支持继续测试更密的时间采样，但不能证明摄影机原生 50 fps，也不能排除局部慢动作或插帧。

Falcao 的 44–57 秒、Kane 的 94–102 秒、Cuadrado 的 136–144 秒已切成 1080p50 试验片段。切片不缩放，使用 libx264 CRF10 有损重编码；六个首尾帧与原片对应帧比较的 PSNR 为 42.26–50.21 dB。粗片段边界来自关键帧查看，不是真实触球标签。新录像的剪辑和时钟不同，旧视频的帧号、模型缓存和候选不能直接搬过来。完整记录见[来源检查 JSON](ADDITIONAL_VIDEO_SOURCES_2026-10-13.json)，切片脚本是 [`prepare_additional_video_pilots.py`](../scripts/prepare_additional_video_pilots.py)。FIFA+ 的归档入口仍有跳转和访问限制。

检查补充录像时，先看助跑起点、触球瞬间、第一次扑救和射门结束是否完整。新增片段需要建立与原罚球的对应关系，并分别记录机位、帧率和时间轴。慢动作回放中的一秒不能直接当成主镜头的一秒。

## 可以用于独立检查的数据

[SoccerNet 官方访问表](https://github.com/SoccerNet/SoccerNet#datasets)将下面三个子数据集列为公开数据：

| 数据 | 用途 | 不解决的问题 |
|---|---|---|
| [Tracking 2023](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023) | 球和人物的小目标检出、漏检、轨迹身份切换；[任务说明](https://www.soccer-net.org/tasks/tracking)提供 1080p、30 秒片段信息 | 不是这 39 次罚球的触球脚标签 |
| [Calibration 2023](https://huggingface.co/datasets/SoccerNet/SN-Calibration-2023) | 用球场线和球门标注检查标定；[官方实现](https://github.com/SoccerNet/sn-calibration)可供比较 | 地面标定不能直接还原空中球的三维轨迹 |
| [Game State Reconstruction 2024](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024) | 检查人物在地面俯视坐标中的定位；[官方实现](https://github.com/SoccerNet/sn-gamestate)提供完整流程 | 不能替代完整助跑和扑救动作的观察 |

整场 `raw_HQ` 和 Ball Action Spotting 数据另有账户与 NDA 要求，见[官方数据说明](https://www.soccer-net.org/data)。公开子数据与受限整场视频是不同的资源。Tracking 的训练压缩包约 9.58 GB，本次只查询了文件清单，没有下载。

## 优先做的改进

先检查原完整录像中的回放，再判断是否需要另一份同场视频。左右脚应围绕鞋、球和触球瞬间做局部跟踪，区分射门脚摆动与支撑脚回收；已有连续姿态帧却判断失败的 Kane、Rashford 片段值得先处理。

回放速度需要同一动作在不同机位之间的对齐。当前 320 个镜头区间中，301 个没有进入有效对齐，另外 19 个进入后未获得足够支持。应先修人物身份和机位关联，再判断回放速度。

扑救高度可以定义成可解释的图像相对量或低、中、高类别，课程没有要求必须使用米。球的最终位置也需要按射门结束来定义，不能自动等同于最后一次检测，更不能把“静止至少半秒”当成课件规定。两项都应保留实际观察和未观察部分。

最后选取新比赛做独立检查。选择标准、训练与测试的划分、参数以及评价方式应在看结果前确定。新比赛能检验泛化能力，同场更清楚的副本则只是增加观察证据；两者不能混成一个准确率。
