# IND5003 Vision II：点球视频分析

这是 IND5003 Vision II 的点球视频分析项目，使用 2018 年世界杯四场点球大战的 39 次罚球：哥伦比亚对英格兰、克罗地亚对丹麦、西班牙对俄罗斯，以及俄罗斯对克罗地亚。

一脚点球从助跑到扑救只有几秒，但里面有不少值得比较的细节：球员走了多少步、用哪只脚触球、门将什么时候开始移动。项目尝试把这些细节从转播画面里提取出来，再看看它们和射门结果有没有关系。代码在小组原项目的基础上继续修改，当前版本重新处理了全部 39 次罚球。

## 查看结果

| 文件 | 内容 |
|---|---|
| [报告 PDF](outputs/fresh_course_experiment/report_v1/REPORT_FRESH_REVIEW_DRAFT.pdf) / [Word](outputs/fresh_course_experiment/report_v1/REPORT_FRESH_REVIEW_DRAFT.docx) | 方法、分析和遇到的问题，目前为审阅稿 |
| [结果表](outputs/fresh_course_experiment/final_bundle_v1/features_39.csv) | 每次罚球一行，共 39 行 |
| [完整 JSON](outputs/fresh_course_experiment/final_bundle_v1/features_39.json) | 各模块的输出、帧号及来源记录 |
| [逐段查看](outputs/fresh_course_experiment/review_v1/index.html) | 下载仓库后在浏览器打开，查看检测图片和对应结果 |
| [统计图](outputs/fresh_course_experiment/final_eda_v1/) | 特征覆盖、步数比较和赛果分析 |

下面是 Kane 罚球片段的检测输出，图中保留了视频帧号，方便回看原画面。

![Kane 罚球片段的检测输出](outputs/fresh_course_experiment/review_v1/evidence/col_eng_p02_kane_eng9_evidence.jpg)

## 怎么看当前结果

视频按罚球切片后，用 YOLO11 提取人物姿态和足球位置。脚本根据球从静止到离开罚球点的过程估计触球时刻，再计算助跑步数和门将动作。现在有 33 次罚球给出了触球时间候选，32 次给出了步数候选；门将触球前移动和扑救方向分别有 29 次、23 次输出。具体数字都在结果表里，遮挡和判断冲突的片段保留为 `unknown`。

现在看，结果还很不均衡。我最不放心的是左右脚，只得到 8 次候选。离球最近的脚可能是支撑脚，射门后的摆腿和支撑脚回收也会混在同一个运动窗口里。Kane 和 Rashford 的片段都有连续的姿态帧，仍然没能通过脚别判断。这说明光把人物检测得更清楚还不够，触球前后的动作要分开看。

步数里，Rashford 的 15 次和 Pivaric 的 13 次让我有些犹豫。它们在三个阈值设置下都得到相同数字，但算法目前数的是脚踝信号的极值，还不能确认每个极值都对应一次落地。可能有连续的小碎步，也可能把一次摆动重复计数。看到这种数字，还是得回到原画面逐帧检查，不能只因为几次运行都一样就认为数对了。

按赛果分组，有步数输出的 20 次进球中位数是 5，12 次未进球中位数是 5.5。这个差别不大，而且两组缺失的片段不一样，步数本身也还没核实。现阶段据此讨论哪种助跑更容易进球，结论会太早。[步数比较图](outputs/fresh_course_experiment/final_eda_v1/04_fresh_approach_outcomes.png)保留了这部分观察。

进球这一列看起来最完整：从转播比分板的红绿图标读出了 35 次结果，与 FIFA 技术报告的记录一致。不过球轨迹规则本身只对 10 次罚球作出了判断。Koke 这一球尤其说明问题：球轨迹给出了进球候选，比分板显示未进，两者冲突，合并结果就保留了未知。比分板能补赛果，却不能证明球已经跟踪正确。35/35 是这批调试视频上的一致性，还没有在新比赛上验证。

俯视图目前只有 32 次罚球的部分地面投影点，不能连成完整的助跑路线。扑救高度只有 6 段局部图像信号，完整扑救的分类还没做好。慢动作和球的最终位置也没有可用的自动结果。门将方向按画面左、右记录，步数只统计可见的助跑部分。这些都是后续需要解决的具体问题。

## 还能补什么

下一步最有价值的是同一次罚球的完整转播和回放。正面机位看不清触球脚时，侧面回放可能有帮助；助跑被切掉一段时，完整录像也比继续调整步数阈值更有用。同一球换了镜头，必须先确认人物和动作对应，再比较它们的时间和位置。

课程说明鼓励寻找额外数据来补充证据和背景。这里仍保留指定的四场比赛作为主分析，其他比赛可以用来检验方法在新画面上的表现。已查到的 FIFA 官方录像和 SoccerNet 辅助数据整理在[补充视频说明](docs/ADDITIONAL_VIDEO_SOURCES.md)里；目前还没有把这些素材加入结果表。

## 在本机运行

查看报告、结果表和检测图片不需要 GPU。重新生成统计图可以直接使用仓库里的结果文件：

```sh
git clone https://github.com/Gabric-cau/ind5003-vision-ii.git
cd ind5003-vision-ii
python3 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python -m src.fresh_eda \
  --bundle-dir outputs/fresh_course_experiment/final_bundle_v1 \
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
