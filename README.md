# IND5003 Vision II — Football Penalty Analysis

## GitHub 完整分析文件下载（2026-10-11）

GitHub 项目地址：<https://github.com/RX422040/vision-ii-penalty-analysis>。
仓库保留源码、测试、配置、Notebook、文档、标注来源记录和选定的检测缓存/实验结果。
另外，`project_files/visionii-support-01.zip`、`project_files/visionii-support-02.zip`
通过 **Git LFS** 保存此前未上传的模型权重、FFmpeg、完整历史输出/缓存/报告媒体、
Notebook 存档及遗漏的 SAM2 辅助文件。两包合计约 **3.58 GB**，还原后新增文件约 **4.69 GB**。
`docs/git_supplement_manifest.json` 记录全部 26,892 个补充文件及 SHA-256。
这批文件来自已校验的完整工程导出，补齐当前及历史分析的文件；不代表已在新电脑逐项重跑全部历史脚本。

**原始视频仍由项目成员单独提供**：把已共享的切片原文件放到 `data/raw_clips/`，
保留文件名及原视频字节。历史工程有 49 个新旧切片，其中当前批量分析使用 39 个 canonical 切片。
重新切片才需要另外提供 `data/full_videos/`。虚拟环境、IDE 配置和临时工作目录不随仓库分发。
仓库是私有的，队友须先获得仓库访问权限。

### 克隆并还原（Windows / PyCharm）

先安装 [Git LFS](https://git-lfs.com/)，再运行：

```powershell
git lfs install
git clone https://github.com/RX422040/vision-ii-penalty-analysis.git
cd vision-ii-penalty-analysis
git lfs pull
python -X utf8 restore_project_files.py
```

还原脚本只使用 Python 标准库，先校验两个压缩包，再解压到项目同名目录，
最后校验每个补充文件。它会复用字节一致的现有文件，默认拒绝覆盖已经改动的文件。
末尾出现 `"all_file_hashes_match": true` 表示补充文件齐全；再次核对可运行：

```powershell
python -X utf8 restore_project_files.py --check
```

如果提示只下载了 Git LFS pointer，请安装 Git LFS 后重新执行 `git lfs install`、`git lfs pull`。
仅点击网页的 **Code → Download ZIP** 不保证包含 LFS 实体文件，优先使用上面的克隆命令。
Git LFS 下载受仓库账号额度限制；不要把只有百余字节的指针文件当作真正的压缩包。
如果遇到 LFS 下载额度限制，可从同一仓库的
[分析补充文件 Release](https://github.com/RX422040/vision-ii-penalty-analysis/releases/tag/v2026.10.11-assets)
下载两个 ZIP，放进克隆项目的 `project_files/` 目录，替换指针文件后执行同一个还原命令。
两包各自是完整 ZIP，不需要合并。
Git LFS 缓存、工作目录中的压缩包、还原文件及源码共需约 13 GB 磁盘空间，Python 环境及原视频另计。

随后在 PyCharm 中打开克隆目录，配置解释器，安装依赖，放好 raw clips，并检查：

```powershell
python -m pip install -r requirements-pycharm.txt
python -X utf8 run_project.py check
python -X utf8 run_project.py replay-ball --output-dir outputs/my_ball_run
```

**迁移边界**：部分历史脚本/缓存记录了原机器绝对路径，运行旧实验时仍可能要改为队友的路径。
`requirements-pycharm.txt` 是实际使用过的版本记录，尚未在全新环境验证完整安装。
历史 OCR 等实验另见对应的 `requirements-jersey.txt` 等文件；SAM2 源码通过普通 Git 和补充包一起保留。
本次上传的 FFmpeg 是 Windows x64 版，其他系统需要替换运行工具。
本次已验证：新克隆目录还原 26,892 个补充文件且全部 SHA-256 一致；
补齐另行共享的原切片后，使用原电脑已有 Python 环境完成 `check` 和 39 段缓存球事件回放，
72 项球模块/路径测试通过，39 段回放共 1,433 帧全部解码。
另外，5 项还原脚本测试通过。以上验证没有重新运行全部 YOLO 推理，也没有验证全新依赖环境或全部历史实验。
验证记录见 `outputs/pycharm_export_validation_2026-10-11/git_clone_restoration_validation.json`。
当前结果仍是实验候选，不能把输出覆盖率、测试通过数或 AI 画面判断写成已验证准确率。

## PyCharm 运行与完整工程导出（2026-10-11）

本次完整工程包为 `deliverables/VisionII_Full_Project_2026-10-11.zip`。
包内包含全部 `src/`、`scripts/`、`tests/`、配置、Notebook、作业文档、原比赛视频、
罚球切片、4个模型权重、当前及历史结果、FFmpeg，以及SAM2源码和辅助模块。
ZIP不包含原机器的 `.venv`、IDE配置、Python字节码、临时目录和已经生成的旧交付压缩包；
这些原文件仍在原工程中。完整视频和历史结果较大，请完整解压后再运行。

当前球事件入口是 `run_project.py`。以下说明优先于下方历史版本的命令和统计。
旧Notebook、旧CSV中的外部赛果标签，与本轮自动规则及AI画面判断分别保留。

### 1. 在PyCharm中打开

1. 完整解压ZIP，打开其中的 `VisionII_Full_Project_2026-10-11` 文件夹；
   应能直接看到 `run_project.py`、`README.md`、`src/` 和 `data/`。
2. 在PyCharm的项目Python解释器设置中选择Python环境。本次实际验证的运行环境为
   Windows x64、Python **3.14.7**。原电脑可选择
   `C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe`；入口会在同版本下追加
   原项目 `.venv/Lib/site-packages`，以使用已有Torch/Ultralytics。
   原项目 `.venv/Scripts/python.exe` 可以运行球模块，但缺少Pandas等部分全项目依赖。
3. 若在新目录/新电脑运行，创建自己的虚拟环境，并在该项目的PyCharm Terminal安装依赖：

   ```powershell
   python -m pip install -r requirements-pycharm.txt
   ```

   该文件记录已运行环境的版本；本次没有声称已在全新环境中成功安装。
   不要复制旧 `.venv` 到新目录。SAM2源码和辅助模块已经随包保留，无需重新下载权重。
4. 新建Python运行配置：

   | 配置项 | 填写内容 |
   |---|---|
   | Script path | 项目根目录的 `run_project.py` |
   | Working directory | 项目根目录，或 `$ProjectFileDir$` |
   | Python interpreter | 第2步选定的解释器 |
   | Parameters | 首次填 `check` |

5. 点击Run。末尾出现 `READY: 39 clips...`，说明球模块所需输入和依赖已找到。
   缺少依赖时，入口会打印模块名；可选全项目依赖缺失会单独列出。
   不填Parameters也默认运行 `check`，不会直接启动长时间模型推理。

### 2. 先看结果，再重跑

```powershell
python -X utf8 run_project.py check
python -X utf8 run_project.py report --open
python -X utf8 run_project.py test-ball
python -X utf8 run_project.py replay-ball --output-dir outputs/my_ball_run
```

在PyCharm运行配置的Parameters栏填相应内容，例如
`replay-ball --output-dir outputs/my_ball_run`。

`replay-ball`使用已经保留的39段检测缓存，重新执行短窗口球关联、门将反弹分支、
门框位置规则、39段慢放导出和文件验证；**不会重新运行YOLO检测**。
它先运行球模块与路径迁移测试，再生成新目录；已有输出目录会拒绝覆盖。
运行结束打开 `outputs/my_ball_run/report.html`。

新目录的结果包括：

| 文件 | 内容 |
|---|---|
| `report.html` | 39段总表、理由、筛选、慢放和候选原像素 |
| `results/*.json` | 自动关联与规则建议，保持独立于画面判断 |
| `outcomes.csv` | 39段AI画面判断与自动规则的并列表 |
| `review_summary.json` | 两种结果的统计、冲突和来源哈希 |
| `visual_assessment.json` | 已有AI原画面判读记录，仅用于展示 |
| `protocol.json` | 本轮参数、源码哈希及新推理/复用数量 |
| `source_snapshot/` | 本轮执行代码快照 |
| `validation_tests.log` / `validation_report.json` | 测试、文件及回放解码验证 |
| `pycharm_run.json` | 解释器、运行模式和迁移运行信息 |

球缓存中保留历史绝对路径作为来源记录；运行时优先解析当前项目
`data/raw_clips/<clip_id>.mp4`，不会因为旧工程仍存在而读回旧工程视频。
新视频必须与原画面判读记录的源文件哈希一致，已有AI判断才会附在报告中。

### 3. 重新运行门前球检测

```powershell
python -X utf8 run_project.py refresh-ball --output-dir outputs/my_ball_fresh_run
```

此模式对39段门前短窗口重新运行YOLO人物/球检测（CPU，imgsz=960），
再执行与 `replay-ball` 相同的关联、规则和导出。
新检测缓存另存为 `outputs/my_ball_fresh_run_fresh/`。
CPU运行比缓存重放慢，终端会逐段显示进度。

这一模式仍复用已经保存的自动主镜头、门框提案、静止种子与来球轨迹。
它不是把所有历史处理步骤都从原比赛视频重新计算。
旧版从原片执行的Notebook和各模块脚本也已包含在完整工程中。

### 4. 其它模块的入口

```powershell
# Notebook环境；打开 notebooks/vision2_penalty_analysis.ipynb
python -X utf8 run_project.py notebook

# 基础39段步数模型处理；新CSV/诊断写到新目录
python -X utf8 run_project.py steps --output-dir outputs/my_steps_run

# 原片机位时间段处理
python -X utf8 run_project.py camera --output-dir outputs/my_camera_run

# 39段SAM2门将完整首次扑救及阶段高度实验，CPU耗时较长
python -X utf8 run_project.py goalkeeper --output-dir outputs/my_keeper_run
```

`steps`为基础计数入口，不能将其输出称为晚间步伐优化或人工确认版。
步伐分类、门将方向与其它实验的专用参数见下方对应版本说明。
历史脚本可能覆盖默认输出，不要无差别批量运行全部 `scripts/`。
以上统一入口为各模块选择新输出目录。

### 5. 当前球结果的边界

当前报告：`outputs/ball_goal_all_events_v2_2026-10-11/report.html`。
AI逐段原画面判断为 **26进球、13未进**；自动规则为 **20倾向进球、6倾向未进、13待定**。
第22、25、30段存在自动规则与AI画面判断冲突，报告保留这些错误，不用画面判断改写自动结果。
门框外、入网前的明显反弹作为未进线索；进网后的反弹不改变进球判断。

“AI画面判断”不是人工真值，“倾向进球”不是已确认球整体越过球门线。
39段解码、测试和文件验证完成，不代表识别准确率已经验证。
重放只是重新计算自动规则；不会自动产生一次新的AI或人工画面复核。

### 6. 导出与校验

本次已在含中文和空格的新目录中实际执行 `replay-ball`：72项测试通过，
39段回放共1433帧全部解码；全部输入视频来自新目录，自动规则结果与发布版一致。
原电脑完整环境导入、SAM2辅助模块和JupyterLab导入也已检查。
证据保存在 `outputs/pycharm_export_validation_2026-10-11/`。
这次验证使用现有解释器，不等于在全新电脑完成依赖安装或全项目重新推理。

```powershell
# 已解压包的完整文件哈希检查，仅需Python标准库
python -X utf8 verify_export.py

# 在原工程中重新生成完整ZIP，拒绝覆盖已有ZIP
python -X utf8 scripts/export_full_project.py --output deliverables/VisionII_Full_Project_new.zip
```

ZIP内 `EXPORT_MANIFEST.json` 记录全部文件、大小与SHA-256；原视频也在清单中。
`verify_export.py` 会读取所有文件，包含大视频，因此比环境检查耗时。
后续自行修改源码/README会产生合理的哈希差异；请保留最初校验日志。

## SAM 2 门将接入（2026-10-11，v4 候选结果）

新版门将结果位于 `outputs/goalkeeper_sam2_update_2026-10-11/results_v4_release/`，
先打开该目录的 `report.html`。`goalkeeper_results_39.json` 和 `results_39/` 汇总全部39例。
剩余6条 unknown 已实际测试；Cuadrado、Smolov、Golovin 得到画面向左候选，unknown 变为3条。
这是候选覆盖率变化，不是准确率；原33条方向、39条高度和整条QA不因方向恢复而提升。
历史v3与旧入口保留，使用下面的新入口执行 SAM2，不会覆写历史结果。

```powershell
python -X utf8 scripts/run_goalkeeper_sam2_update.py --head-part-fallback --output-dir outputs/my_sam2_run
```

默认SAM2输入1024；`--image-size 512` 可用于CPU速度实验，但并非等效精度。
本次另5例用512；Cuadrado的512身体掩膜导致头部重叠不稳定，因此发布结果保留1024身体、
经模型/裁剪像素校验的512头部缓存。关键点不够时，自动鼻子正提示、肩/髋负提示初始化
SAM2头部子区域；头部跳变处停止，不把后续漂移当证据。
`--clips` 可限定旧v3的 unknown 清单；未传时处理6例。`--resume` 要求协议和输入哈希一致。
本次仅Cuadrado运行头部fallback，不能宣称另5例的头部fallback也已验证。

依赖见发布目录的 `requirements-sam2.txt`；官方SAM2辅助包隔离在
`.deps_goalkeeper_segmentation`，没有更新原来的Torch/OpenCV环境。
84项所选测试通过、6个复核视频290帧解码通过；AI静态复核不是用户人工真值。
独立身体掩膜不能决定左右，方向须结合头/躯干或自动头部子区域与持续侧向转变。
坐标仍是画面左右，不自动换算守门员自身左右。

## 球轨迹连续性修正（2026-10-11，实验）

上一轮把旧版完整飞行段缩成5个起始点，再用全片得分重新关联，
导致 Iniesta 第65～67帧被门将附近的误检替换。现在完整保留此前自动跟上的飞行段，
只在末端续接；预测依据最近运动趋势更新，允许图像轨迹逐渐弯曲。
新增关联在2个原始帧后固定，后面的长串误检不能改写这些观测。
突然转向的接近判断只看此前球的位置，新候选落在门将身上不能为自己制造碰撞理由；
当前飞行仍有连续候选支持时优先续接。几何接近仍不是物理碰撞证明。

修正回放：`outputs/ball_continuity_preserved_2026-10-11/report.html`；
第67帧证据：同目录 `esp_rus_p01_iniesta_esp6_frame_67.jpg`。
绿色是保留及续接的观测，蓝色是此前轨迹，红色叉是上一版失败轨迹，橙色是预测。
四条样本均保留完整旧飞行段，Iniesta 的18个已有观测不再被覆盖。
这解决了删除已有连续轨迹的回归，完整后段跟踪及真实识别准确率仍未验证。
原39条结果没有替换，预测不计为检测、碰撞或越线证据。

```powershell
python -X utf8 scripts/run_ball_continuity_probe.py --fixed-incoming --collision-gated --comparison-dir outputs/ball_continuity_2026-10-11_fixed_incoming --regression-dir outputs/ball_prediction_collision_2026-10-11_v2 --output-dir outputs/ball_continuity_replay_new
# 保留完整已有飞行段；不要用 --prefix-points 5 缩短它。输出目录必须是新目录。
```

## 新增实验：门将/裁判身份过滤（2026-10-11）

入口为 `scripts/run_goalkeeper_role_experiment.py`，身份模块为 `src/goalkeeper_role_association.py`。
先按身体关键点分人，再用触球前多帧脚踝到门线距离（60%）、身体在球门区域（20%）、
手腕姿态（10%）与可见性（10%）选择门将候选；排除其他人物后进行有断轨限制的跟踪。
手腕是辅助证据，不是手套识别；权重为工程设定，不是身份概率。

Cuadrado片段已能保留门将、排除前景裁判，但门将横卧并部分遮挡后仍漏检，方向仍unknown。
模型放大、旋转与局部人物ROI实验未解除这一例；旋转版还出现了身份歧义。
39条身份筛选中36条给出候选，起始人物与旧版相同；这不是36/39的准确率。
原v3方向总表未覆盖，仍33条方向候选、6条unknown。74项相关测试通过。

报告、标注视频、逐帧排除记录与源码快照：
`outputs/goalkeeper_role_association_2026-10-11/final_v4/report.html`。

```powershell
python -X utf8 scripts/run_goalkeeper_role_experiment.py --all-roles --fresh --output-dir outputs/goalkeeper_role_new_run
# 输出目录必须是新目录。现有本地YOLO11m模型；无需下载新权重。
```

## 最新门将扑救方向：姿态优先 v3（2026-10-11）

加入直立到侧扑的身体姿态变化：读取同一选中轨迹的双肩、双髋和双踝，
连续身体轴线支持侧向伸展时，优先于检测框中心移动方向；不用固定横移距离作为必要条件。
仍保留置信度、连续性、身份歧义和断轨检查；没有充分姿态证据时沿用v2位移分支。
39条重算：unknown由上一轮11条降至6条，另修正4条旧方向；
33条方向候选中画面左16、画面右15、center2。Kane输出画面左，在用户确认的机位中为自身右侧。
高度标签39条未改变，原完整窗口QA接受子集仍15条；49项相关测试通过。
未将用户方向答案作为推理输入；静态AI检查不冒充人工真值，候选覆盖率不等于准确率。

当前报告：`outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final/report.html`；
逐片结果及原帧证据在同目录 `clips/` 与 `visual_qa/`，汇总为 `goalkeeper_results_39.json`。

```powershell
python -X utf8 scripts/run_goalkeeper_direction_update.py --resume
python -X utf8 scripts/review_goalkeeper_pose_direction.py --output-dir outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final
# 默认为pose_direction_v3；源码或参数改变时使用新输出目录。
```

## 历史门将方向逻辑 v2（2026-10-11）

方向与完整扑救观察分开验收，入口为 `scripts/run_goalkeeper_direction_update.py`。
39条缓存重算后，方向未知由23条降至11条，新增12条早期侧向运动候选；
合计28条方向候选。高度仍按完整1秒观察窗口判断，整体门将QA子集仍15条。
Henderson恢复为画面向右；在该主镜头中对应门将自身向左。
结果与39条对照：`outputs/goalkeeper_direction_2026-10-11/report.html`；
逐片结果：同目录 `clips/`；新逻辑29项相关检查通过。
这些是缓存轨迹支持的候选，不是新增人工真值或已测识别准确率。
下方2026-10-09及旧交付包的门将统计属于历史版本。

```powershell
python -X utf8 scripts/run_goalkeeper_direction_update.py --policy separate_direction_v2 --output-dir outputs/goalkeeper_direction_v2_replay
# 原v2结果已冻结；旧输出目录中的protocol保留了当时源码哈希。
```

## 最新实验：不完整模块优化与验收（2026-10-09 晚间）

本轮实验结果独立保存在 `outputs/optimization_evening_2026-10-09/`，没有覆盖旧结果或原视频。
这是实验候选版本，不是已验证的最终模型。候选覆盖率、规则 QA、参数稳定性均不等于准确率。

- 步数候选从 31/39 增至 39/39；原有31条中28条不变、3条变化1步。
  Brozovic 的2步仅覆盖可见的后段助跑，不能解释为完整助跑只有2步。
  连续助跑候选13条、末段调整候选4条、类型未定22条。
- 左右脚的名义输出有39条，但固定触球锚点/窗口敏感性检查只有1条同时满足一致性及优势比要求；未作为准确率提升采用。
- 守门员由球门区域自动播种，不读人工/AI身份种子表；使用ROI姿态、外观一致性与断轨限制。
  外观版29/39通过原QA，加入完整1秒观察窗口检查后15/39通过；两种口径不同，不能解释为准确率升降。
  对后半段丢失的扑救输出unknown，保留未过滤的候选以便复查。相似球衣/裁判仍可能混淆。
- 球的13条困难样本进行4组对照；连续性QA为全图32帧7条、全图60帧6条、ROI32帧7条、ROI60帧5条。
  其中1条无可用触球锚点而弃权，26条原通过样本本轮未重新推理；不能把两批相加宣称新准确率。
- 39条标准视频重新分段，得到187个连续、无遗漏的候选区间；四类机位与慢动作仍未验证。
  球门竖直平面候选不等于球场地面单应性；自动outcome、真实落点/球速仍未解决。

入口：

- 报告：`output/pdf/vision2_optimization_experiments_2026-10-09.pdf`。
- 实验明细与复现：`outputs/optimization_evening_2026-10-09/README.md`。
- 真实39条联合结果：`outputs/optimization_evening_2026-10-09/integration/canonical_results_39.json`。
- 实际环境快照：`requirements-evening-experiment.txt`，不是已通过全新环境安装的锁文件。

旧 `data/processed/penalties_features.csv` 的49行来自示例流程，**不能作为本项目实证总表**。
本轮 EDA 只读取标准39条实际输出，外部赛果标签单列，不冒充自动预测或用户人工复核。

```powershell
# 在项目根目录；沿用此次已运行环境，不替换用户已有环境。
$visionPython = 'C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe'
& $visionPython -X utf8 scripts/vision2_evening.py inspect
& $visionPython -X utf8 scripts/vision2_evening.py validate
& $visionPython -X utf8 scripts/vision2_evening.py bundle
```

本轮168项选定测试、39份步数重放与39个原视频哈希检查通过，不等于真实识别准确率。
此前版本与原Notebook仍保留；下文是历史说明，不代表晚间实验输出已回写进原Notebook。

## 历史版本：按单次罚球方式解释步数（2026-10-09）

现在步数分析明确分为三种呈现方式：

| 本次罚球方式 | 步数处理 | 不做什么 |
|---|---|---|
| 连续助跑候选 | 整体计数，触球保护区另列 | 不强行划出最后三步/五步 |
| 末段调整候选 | 保留总数，分列末段前、末段及保护区 | 不因为碎步就减掉或合并候选 |
| 暂不能判断 | 总数可用就保留，类型与分区未定用null | 不把类型未知写成0步或提取失败 |

“连续”是可见候选节奏相对连续的描述，并不意味着大步、冲刺或完全没有调整；
整体均匀的密集小步也可能属于此类。类型作用于一次罚球，不是某球员的永久标签。
直线/弧线仍是独立路线维度；不根据类型、姓名、进球结果或用户粗看步数修改峰谷阈值。

- 新复核页：`http://127.0.0.1:8882/steps/index.html`（本地服务器运行时）。
  支持按方式筛选、整体/分段计数、参数范围与原/增强姿态对照。开启盲看会取消类型筛选。
- 结果：`outputs/style_aware_steps_2026-10-09/classified/`，对照为`classified_refined/`。
  本次基础结果14条连续候选、2条末段调整候选、23条类型未定。
  类型未定中15条仍有总候选数、8条继承上游无可靠计数；31条已有总数全部保留。
- 新模块：[style_aware_steps.py](src/style_aware_steps.py)，批处理：
  [run_style_aware_steps.py](scripts/run_style_aware_steps.py)。
- Notebook入口：[vision2_step_style_review.ipynb](notebooks/vision2_step_style_review.ipynb)。
  它只读取新结果，不覆盖约49MB的原`vision2_penalty_analysis.ipynb`，不写人工标签。

### 连续型不是“没找到变化”的同义词

至少两个有效候选间隔；安全助跑段姿态支持率≥90%、直接观测率≥80%；
间隔的MAD/中位数≤0.20、最大间隔/中位数≤1.80、全距/中位数≤1.0；
末段尾部与候选主方向可用；没有停顿候选；三套峰谷阈值、三套变化规则和
三个触球锚点对照完整且均无末段变化候选。缺少这些证据时保留“类型未定”。
MAD是绝对偏差的中位数，不是准确率。这些阈值是公开的、未验证的工程选择。

短助跑可能只有3–4个安全候选，足以描述相对连续的节奏，却不满足旧变化点模型
“前后各两个间隔”的要求；这两种判断分开保存，并明确提示不是已证明没有调整。
默认分析发现末段变化时保留调整候选，同时显示参数分歧，不冒充确定分类。

计数分区互斥且守恒。主数是名义触球前的助跑候选，仍包含单列的触球保护区，
因为最后真实支撑脚与摆腿峰谷尚未分清。助跑前背景动作单列，不并入主数。
所有参数计数统一使用`起点 <= frame < 触球边界`，输出3×3阈值/触球锚点表及
最小—最大范围，不将其称为统计置信区间、人工允许误差或真实落脚范围。
每条新增`step_analysis`；旧`footwork_result`、原事件、原总数及原视频保持不变。

### 最新复现与检查

```powershell
python scripts/run_style_aware_steps.py
python scripts/build_penalty_phase_review.py --classified-dir outputs/style_aware_steps_2026-10-09/classified --refined-classified-dir outputs/style_aware_steps_2026-10-09/classified_refined --output-dir outputs/phase_optimization_2026-10-09/review/steps --data-only --media-prefix ../media/ --comparison-path outputs/style_aware_steps_2026-10-09/model_comparison.json
python scripts/validate_style_aware_steps.py
python scripts/serve_penalty_phase_review.py
```

无新增依赖，沿用`requirements-phase-experiment.txt`；运行使用项目已有Python环境。
本次146项选定测试通过（新增22、末段26、此前98），78份结果通过输入保留、
计数守恒、数值重放与哈希检查。39份已完整解码的播放副本再次核验源/副本哈希。
这些检查不等于真实视频准确率；所有结果仍未经逐帧人工确认。
验证记录见新结果根目录的`validation.json`与`browser_qa.json`。
Notebook代码已在Python中只读执行并产生39行表格，未改写Notebook执行输出。

## 历史：路线与末段调整分开分析（2026-10-09）

按最新讨论，前段弧形/横向路线是背景描述，临近触球的节奏变化是另一个维度。
一个球员可以同时有弧形路线和末段调整；不再把这些当作互斥的球员类型。
当前改动只重算39条视频的缓存证据，不重新训练，也不改变原有姿态、触球锚点、
助跑总候选数、goalkeeper、ball或outcome结果。

- 新版复核页：`outputs/phase_optimization_2026-10-09/review/terminal/index.html`。
  启动下方本地服务器后访问 `http://127.0.0.1:8882/terminal/index.html`。
  视频沿用完整、已解码验证的H.264副本，不重复转码。
- 新结果：`outputs/terminal_footwork_2026-10-09/classified/`；
  姿态对照：同级 `classified_refined/`。各有39个JSON与`summary.json`，
  其中实际使用增强姿态的配对仍为30条，其余是原姿态回退，不能算增强实验。
- 主模块：[terminal_footwork.py](src/terminal_footwork.py)；
  批处理：[run_terminal_footwork.py](scripts/run_terminal_footwork.py)；
  测试：[test_terminal_footwork.py](tests/test_terminal_footwork.py)。

### 新方法与统计口径

1. 沿用独立的人物—球关联、触球窗口与助跑起点，不使用用户粗看步数调参。
2. **触球保护区单列**：安全分析截止为“触球窗口左端”和“名义触球前0.12秒”
   两者中更早的位置。保护区的候选仍保留在总数中；它可能是真实最后支撑落脚，
   也可能是摆腿峰谷，本算法不假装已经分清。
3. 在安全截止前最多1.6秒内搜索变化点，变化点前后各至少两个连续有姿态支持的
   候选间隔。比较间隔时长及经镜头补偿后的局部髋部位移；不规定最后几步必属末段。
4. 默认节奏压缩比不超过0.78，伸长比至少1.45，绝对变化至少0.06秒；
   局部推进量缩小比不超过0.65，绝对变化至少0.06个人物框高。
   变化须覆盖至少67%的末段间隔，且前后两段的中位绝对偏差之和比单段降低至少25%。
   前段最多参考5个间隔，末段尾部距离安全截止不超过0.55秒，长缺帧不跨越。
   这些都是未用真值拟合/验证的工程阈值，不是生物力学标准或概率。
5. 髋部推进使用相邻候选间的二维局部位移，不投影到固定全局方向；
   弧线转向本身不应被当成“步长缩短”。镜头补偿不完整或尺度变化大时，禁用路线与
   推进量判断，仅在条件足够时保留时间间隔分析。图像髋部位移仍不是真实脚步长度。
6. 分别报告“末段变化候选”“未发现清晰变化”“证据不足”；未发现变化不等于没有
   调整，证据不足不等于0步。变化点对应的共同端点候选默认归末段，同时列出去掉
   该候选后的计数，公开这一候选的边界歧义。
7. 三种事件显著度、三种变化规则和早/名义/晚触球锚点分别重算，显示分歧。
   保留增强姿态对照，不自动选数量更接近预期的版本。阈值范围不是置信区间。

`footwork_result`新增路线、末段、计数分区、逐候选归属、间隔特征和敏感性；
旧`phase_result`、`event_proposal`、`count_sensitivity`原样保留。
分区互斥，名义助跑总候选 = 末段前 + 末段 + 未发现分界 + 证据不足未分配 +
姿态缺失 + 触球保护区。`null`表示该类数量未估计，不代表0。
若未检测出分界，路线范围是整个可见助跑，不能冒称已经分出的“前段”。

**脚踝控制不在当前可测范围内。** COCO17只有踝点，没有脚跟/脚尖/足部朝向，
不能据此测量踝角、锁踝刚度、力矩、发力质量或“职业球员角度不重要”。
候选峰谷不是已经验证的落脚；脚踝轨迹只是可观测运动信号。

### 本次结果与复现

基础姿态39条：2条出现末段变化候选（Rashford、Aspas），15条未发现清晰分界，
22条证据不足。其中8条继承上游人物/触球/助跑未解决问题，14条虽有助跑候选，
但不足以进行新的末段前后比较；**不是22条视频提取失败**。
所有可用的名义助跑总候选数保持原值。9条在参数/触球锚点变体间有分类分歧。
Rashford的默认末段边界f86、Aspas为f78，均是候选，不能称为正确分段。

Cuadrado依然是名义5个助跑候选，其中1个进入触球保护区；剩下4个不足以
比较前后各2个间隔，所以不强行给出“前段几步＋末段几步”。敏感阈值与增强姿态
会给出不同结论，页面明确显示。用户“约5步、4–6可接受”没有用作训练标签。

```powershell
python scripts/run_terminal_footwork.py
python scripts/build_penalty_phase_review.py --classified-dir outputs/terminal_footwork_2026-10-09/classified --refined-classified-dir outputs/terminal_footwork_2026-10-09/classified_refined --output-dir outputs/phase_optimization_2026-10-09/review/terminal --data-only --media-prefix ../media/
python scripts/validate_terminal_footwork.py
python scripts/serve_penalty_phase_review.py
```

使用`requirements-phase-experiment.txt`中的现有环境，无新增依赖。当前Windows验证解释器：
`C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe`。
缓存输入与源代码哈希不匹配时停止，不能悄悄混用新旧结果。
本次124项选定测试（新增26＋原回归98）通过，78份结果的原字段保留、计数分区、
边界和哈希检查通过，39份已解码播放副本的源/副本哈希再次核验一致。
机器结构/回归验证见`outputs/terminal_footwork_2026-10-09/validation.json`；
五条视频的浏览器交互检查见同目录`browser_qa.json`及`review_preview.png`，
不等于全39条人工逐步复核，也不构成准确率。

## 历史：动作分阶段分类实验（2026-10-09）

已把准备/调整、起步、助跑接近、踢球附近和随动恢复分开，并增加横移、
弧线、绕圈状投影、提速、停顿和密集低推进量动作的候选描述。
先检查人物—球关联，再分类；不是把旧总步数直接拆成几个数字。

- 验收页：`outputs/phase_optimization_2026-10-09/review/index.html`；
  晨间报告：同目录 `report.html`，包含39条逐视频摘要。
- 完整方法、限制和复现：[实验说明](docs/PENALTY_PHASE_EXPERIMENT_2026-10-09.md)。
- 最新实验结果：`outputs/phase_optimization_2026-10-09/classified/`；
  增强姿态对照：`classified_refined/`。两者不是已经验证的标准答案。
- 39条处理，31条产生暂定阶段、8条拒绝强行分段；独立触球证据为
  20条较强、12条暂定、7条弃权，另外1条因人物轨迹过短不能分阶段。
- 98项选定单元/回归测试与两版78条结构检查通过；39个完整H.264播放副本
  解码、长度和尺寸检查通过。**这些不是准确率**，全数仍 `human_verified=false`。
- 真实可比的增强姿态有30条，其中19条候选数变化、8条相差至少2；
  未证明大模型更准，所以保留对照，未自动替代基础版本。

本地打开：`python scripts/serve_penalty_phase_review.py`，访问
`http://127.0.0.1:8882/index.html`。只监听本机，不上传视频。
页面支持0.25×慢放、逐帧、人物框、阶段时间线、参数敏感性及盲看。
原视频、历史CSV、历史JSON和旧notebook没有被本轮实验覆盖。

**对下文旧版结果的补充纠正：**本轮AI视觉检查全部39张旧版QA图，发现11条
旧轨迹落在门将或裁判。旧版“39/39能输出数字”不表示人物、触球或步数正确。
独立球—人关联及Kovačić后续镜头重选改善了其中部分案例，其余保留不确定。
AI看图属于质检，不是用户人工复核；不要引用历史 `manual_completion` 当作真值。

## 历史：镜头选择修复（2026-10-09，不使用 mask）

步数流程不再默认“第一个镜头就是罚球”。默认 `first_viable` 策略先扫描
视频中的镜头边界，再按时间顺序分别运行姿态识别，选择第一个满足现有
全身姿态覆盖和动作条件的候选镜头。每个镜头重新建立轨迹，所有输出帧号
仍是原视频的零起始帧号。颜色分布变化和空间亮度突变共同辅助检测切镜头；
没有增加草地 mask 或“球必须运动”的门槛。

如果清晰但动作较弱的人物（例如裁判）与动作更强、检测较弱的人物竞争，
程序会触发裁剪放大重试，不再仅因前者关键点覆盖率较高就停止重试。
不足 1 秒的镜头、所选人物框高度超过画面高度 60% 的近景不能直接通过
镜头候选筛选；这些也是启发式规则，不是训练好的动作分类器。

最新运行结果在 **`outputs/shot_repair_2026-10-09/all_clips_final/`**。
最终运行回归：39/39 条生成候选结果，原失败的 4 条恢复输出，未新增运行失败；
原有 35 条中有 9 条候选步数发生变化，不能据此判断准确率提高。
33/39 条仍有质量警告，6 条未触发警告也不等于已验证正确。
30 项单元/回归测试通过；39 条帧号范围、源代码哈希、结果完整性与 QA 图片解码
检查通过，详见 `validation.json`。人物与动作的视觉抽查集中在原失败四条及排错案例，
并未对全部 39 条逐步人工标注。
其中四条原失败视频的结果如下：

| 视频 | 所选镜头帧范围（末端不含） | 自动候选步数 | 验证状态 |
|---|---|---:|---|
| Modrić / Croatia–Denmark | [28, 125) | 6 | 未经用户人工确认 |
| Golovin / Spain–Russia | [33, 160) | 4 | 未经用户人工确认 |
| Ignashevich / Russia–Croatia | [25, 127) | 4 | 未经用户人工确认 |
| Vida / Russia–Croatia | [46, 150) | 3 | 未经用户人工确认 |

每条有完整 JSON 和黄色人物框的 `_qa.jpg`；汇总为 `comparison.json`。
上一级同名文件、`refined/`、`all_clips/` 均为中间试运行，**以 all_clips_final 为准**。
其中最初 Ignashevich 试运行错误选中了裁判。原 `data/processed/penalty_step_counts.csv` 和
`outputs/step_diagnostics.json` 保持不变，仍反映历史版本，不能冒充这次新结果。

这次验证解决的是镜头截取及该例的人物选择故障，不是步数准确率验证。
四条仍标记 `review_required=true`、`human_verified=false`；踢球帧、左右脚、
落脚次数仍可能有误。镜头切分和动作筛选是启发式方法，仍可能遇到切镜头漏检、
只剩回放、助跑不完整、遮挡和关键点抖动；程序不会把后续镜头自动标为人工确认。

```powershell
# 仅重跑历史失败条目，不覆盖原有主表；无需 --field-mask
python scripts/rerun_step_shot_repair.py --output-dir outputs/shot_repair_2026-10-09/failed_clips_rerun

# 全量回归另存，随后检查帧号、完整性与源代码版本；不是准确率验证
python scripts/rerun_step_shot_repair.py --all --output-dir outputs/shot_repair_2026-10-09/all_clips_final
python scripts/validate_step_shot_repair.py

# 完整流程默认采用新策略；保留旧选镜头策略作对照时用 --shot-selection first
python scripts/run_step_counting.py --shot-selection first_viable --output-csv outputs/steps_multishot.csv --diagnostics-json outputs/steps_multishot.json
```

本次在已有 Python 3.14 环境和本地 YOLO11n Pose 权重上运行，未安装新模型、
未增加依赖。回归测试覆盖跳过开头特写、镜头隔离、绝对帧号、候选竞争时的重试、
无可用镜头时的失败保留，以及原有步数、守门员和球特征测试。
实际环境版本记录在 `outputs/shot_repair_2026-10-09/runtime.json`，不等同于
`requirements.txt` 中所有固定版本。本机复现使用已有环境，并补充本地模型包路径：

```powershell
& 'C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe' -X utf8 -c "import sys,runpy; sys.path.append(r'C:\Users\30605\PycharmProjects\JupyterProject\.venv\Lib\site-packages'); sys.argv=['scripts/rerun_step_shot_repair.py','--output-dir','outputs/shot_repair_2026-10-09/failed_clips_rerun']; runpy.run_path(sys.argv[0],run_name='__main__')"
```

## Provenance correction and experimental optimization (2026-10-08)

The legacy files named `manual_reviews` / `confirmed` were created by the
assistant, not reviewed by the user. There are currently **zero independently
human-verified step references**. Prior exact-match, MAE and RMSE numbers were
agreement with AI estimates, not validated model accuracy, and are withdrawn
as accuracy claims. Original files are preserved in
`outputs/archive/provenance_before_2026-10-08/`.

Step exports now retain `candidate_step_count` and leave `confirmed_step_count`
empty unless explicit human provenance is present. Kick-frame and goalkeeper
seed annotations are also marked `ai_assisted_unverified`. Historical JSON
diagnostics remain evidence of earlier runs, not verified labels. Outcome rows
retain their external match-result references; they are separate from visual
tracking validation.

See `docs/vision2_optimization_audit_2026-10-08.md` for the source audit,
HSV-mask experiments, top-down calibration requirements and remaining work.

Run the mask-only ablation on frozen detections without loading YOLO:

```powershell
python scripts/run_mask_ablation.py
```

Live pose inference can enable the experimental field prior with
`python scripts/run_step_counting.py --field-mask`. For goalkeeper stabilization,
use `--field-mask` with `scripts/run_goalkeeper_analysis.py`.
Goalkeeper and ball runners reject unverified annotation inputs by default;
`--allow-unverified-annotations` permits an explicitly exploratory run, never a
ground-truth evaluation. The optional CPU ablation dependencies are listed in
`requirements-mask.txt`. This audit used a local `.deps_mask/` package target
because the original `.venv` refers to a Python installation no longer available.

This repository contains the complete, reproducible workflow for the IND5003
Vision II project. It analyses 39 penalty kicks from four 2018 FIFA World Cup
shoot-outs:

- Colombia vs England
- Croatia vs Denmark
- Spain vs Russia
- Russia vs Croatia

The project extracts the required features for the penalty taker, goalkeeper,
ball and shot outcome:

- kicking foot and run-up step count;
- goalkeeper pre-kick movement, dive direction and dive height;
- ball trajectory and final image position;
- observed `goal` / `no_goal` outcome.

## Repository layout

```text
JupyterProject/
├── configs/                       # Pipeline and canonical clip manifests
├── data/
│   ├── annotations/               # Provenance-labelled AI estimates and external outcome references
│   ├── full_videos/               # Source match videos
│   ├── raw_clips/                 # Canonical and legacy penalty clips
│   └── processed/                 # Final analysis-ready CSV files
├── docs/
│   └── ind5003_vision_ii_project_requirements.pdf
├── models/
│   └── weights/                   # Local YOLO object and pose weights
├── notebooks/
│   ├── vision2_penalty_analysis.ipynb
│   └── archive/                   # Pre-feature notebook snapshots
├── outputs/
│   ├── ball_qa/                   # Ball trajectory contact sheets
│   ├── goalkeeper_qa/             # Goalkeeper identity and movement QA
│   ├── plots/                     # Final analytical figures
│   ├── archive/                   # Retained calibration/debug evidence
│   └── *_diagnostics.json         # Rich reproducibility evidence
├── scripts/                       # Batch runners, QA renderers and notebook updaters
├── src/                           # Reusable analysis modules
├── tests/                         # Unit tests
├── README.md
└── requirements.txt
```

## Primary deliverables

| Category | File | Description |
|---|---|---|
| Notebook | `notebooks/vision2_penalty_analysis.ipynb` | Complete exploratory and scripted workflow |
| Step count | `data/processed/penalty_step_counts_confirmed.csv` | 39 AI candidate counts; human-confirmed fields currently empty |
| Goalkeeper | `data/processed/penalty_goalkeeper_features.csv` | Pre-kick movement and dive features |
| Ball and outcome | `data/processed/penalty_ball_outcome_features.csv` | Ball trajectory summary, final position and outcome |
| Outcome labels | `data/annotations/penalty_outcomes.csv` | Auditable official-result labels |
| Diagnostics | `outputs/step_diagnostics.json` | Retained pose and step-counting evidence |
| Diagnostics | `outputs/goalkeeper_diagnostics.json` | Goalkeeper track selection and normalized trajectories |
| Diagnostics | `outputs/ball_outcome_diagnostics.json` | Candidate detections and selected ball trajectories |

The complete Chinese methodology and step-by-step research workflow is in
`docs/vision2_research_methodology_and_workflow.md`.

For course alignment, third-party attribution, module-by-module code provenance,
and a comparison with the official SoccerNet tasks and repositories, see
`docs/vision2_code_provenance_course_soccernet_mapping.md`.

The current Ball/Outcome export has 39 unique rows and no empty tracks. It
contains 26 `goal` and 13 `no_goal` labels. Twenty-six ball tracks are
automatically accepted; 13 are deliberately marked `review_required` because
their detector coverage or contact anchor is less reliable. Review flags are
preserved instead of replacing uncertain evidence with guessed coordinates.

## Environment setup

Historical setup example. For the current exported project, use the PyCharm
instructions and `requirements-pycharm.txt` at the top of this README.
The currently observed runtime is Python 3.14.7; other versions have not been
validated for all of the pinned packages. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository already contains the required model weights:

- `models/weights/yolo11n_object_detection.pt`
- `models/weights/yolo11n_pose.pt`
- `models/weights/yolo11m_pose.pt`

## Running the workflows

Run every command from the repository root so the relative data paths remain
consistent.

### 1. Penalty-taker step counting

```powershell
python scripts/run_step_counting.py
python scripts/merge_manual_step_reviews.py
```

### 2. Goalkeeper features

The first command is the expensive high-resolution pose refresh. It is only
needed when the videos or model weights change.

```powershell
python scripts/rerun_goalkeeper_pose.py --all --allow-unverified-annotations
python scripts/run_goalkeeper_analysis.py --allow-unverified-annotations
```

Generate goalkeeper QA sheets with:

```powershell
python scripts/render_goalkeeper_qa.py
python scripts/render_goalkeeper_candidates.py
```

### 3. Ball trajectory and outcome

```powershell
python scripts/run_ball_outcome_analysis.py --allow-unverified-annotations
python scripts/render_ball_qa.py
python scripts/render_ball_qa.py --review-only
```

### 4. Notebook

Start Jupyter from this directory, then open
`notebooks/vision2_penalty_analysis.ipynb`:

```powershell
jupyter lab
```

## Validation

The core tests use the standard library's `unittest`; `pytest` is also included
for the function-style goalkeeper tests.

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest tests -q
```

## Method and interpretation notes

- Kick frames are unverified AI visual estimates stored in
  `data/annotations/penalty_kick_frames.csv`.
- Step counts use pose trajectories. AI visual estimates are separate candidates,
  not human review overrides or ground truth.
- Goalkeeper displacement is normalized by body scale and stabilized against
  camera motion where sufficient background flow is available.
- Ball tracking combines YOLO sports-ball candidates, moving-player ankle
  anchors, colour context and motion toward the goal.
- Ball speed is reported in pixels per second. It is not a physical km/h
  estimate and should only be compared cautiously across similar camera views.
- Outcome is an observed label, not inferred from final ball geometry. This
  avoids treating saves, crossbar hits and goals as geometrically equivalent.
- Image-left/image-right describe screen coordinates, not the goalkeeper's
  anatomical left/right.

## Source and archive policy

The assignment brief is retained in `docs/`. Original intermediate notebooks
are retained in `notebooks/archive/`. Calibration experiments are retained in
`outputs/archive/ball_tracking_calibration_debug_2026-09-02/` so the working
root stays clean without discarding evidence.

The video files are source research material. Confirm that any submission or
redistribution complies with the applicable course and media-use rules.

## Goal-facing evidence experiment (2026-10-10)

This is an isolated **review-aid experiment**, not a replacement for the
39-clip baseline and not a validated goal/no-goal classifier. Its outputs live
under `outputs/goal_event_evidence_2026-10-10/`. Original videos and earlier
results are preserved. No outcome annotations are read by these scripts.

The pipeline uses source-video ball detections (full frame plus enlarged goal
ROI), camera-compensated HSV white/motion components, a multi-frame stationary
ball seed near the kicker, and two-point departure confirmation. Near-identical
encoded frames are excluded from motion estimation but retain their original
frame IDs. A supported incoming segment is fixed before a bounded outgoing
segment is searched; this prevents a long glove track from replacing the shot.
An HSV goalkeeper-kit mask, learned from an existing **unverified** pre-kick
keeper candidate, flags ball/body overlap. Neither proximity nor overlap proves
physical contact. The outgoing search is limited to eight distinct frames;
it does not establish where the ball finally rests or when play ends.

The side-by-side videos show original pixels and a goal-plane homography.
This is not a reconstructed front camera: ball/player depth remains unknown.
Missing ball observations are not interpolated. Small trail dots represent
earlier measurements; circles mark exact-frame observations. All automatic
outcomes remain `unknown`, including when the ball projects inside the mouth.

Development clips: Cuadrado and Henderson (`col_eng_p03`, `col_eng_p06`).
Additional frozen-parameter probes: Eriksen and Iniesta (the first canonical
penalty in two other matches). These are not accuracy-validation sets: there
are no independently verified frame-level ball labels. AI visual inspection
must not be described as the user's human review or ground truth.

Reproduce in a new output directory (scripts refuse to overwrite clip results):

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/cache_goal_event_detections.py --batch-size 1 --output-dir outputs/my_goal_probe/detections
python scripts/cache_goal_event_people.py --cache-dir outputs/my_goal_probe/detections
python scripts/probe_goal_event_tracking.py --v2 --cache-dir outputs/my_goal_probe/detections --directory release_v6
python scripts/render_goal_event_video.py --input-dir outputs/my_goal_probe/release_v6 --output-dir outputs/my_goal_probe/videos
python -m unittest discover -s tests -p "test_goal_event_tracker.py" -v
python -m unittest discover -s tests -p "test_goal_plane_rectification.py" -v
```

Dependencies are already included above; no new package or model download is
required. Cached upstream pose/contact/goalkeeper candidates from the October 9
experiment and the existing local YOLO weights are prerequisites. Detection
batch size 1 reduces memory pressure. Output MP4s use H.264/yuv420p/faststart and
are fully decoded after rendering; original source hashes are checked.

Current experiment handoff: `videos_release/` contains two development demos;
`release_v6/` contains their measurements; `additional_probe/release_v6/`
contains the untouched extra-probe results. `assistant_visual_review.json`
records the AI-only review, including incomplete Eriksen coverage and suspected
Iniesta ball/net confusion. `experiment_audit.json` records regression and
artifact checks. Do not present the additional probes as successful full-event
tracks or the automatic outcome module as complete. None of these candidates
have been promoted into the canonical 39-clip outputs.

## Camera timing experiment / 机位时间定位（2026-10-10）

**当前交付是可复现的候选生成与复核工具，不是四类机位均已准确识别。**
独立入口为 `scripts/run_camera_timing_analysis.py`，本轮交付目录是
`outputs/camera_timing_2026-10-10/release/`。旧机位结果、第一轮实验、原始
39 段视频以及此前球/守门员/步数结果均保留，不把候选写回人工标注表。

### 方法、来源与输出边界

| 阶段 | 做了什么 | 方法/代码来源与限制 |
|---|---|---|
| 输入与时间轴 | 只读取切片清单中的 39 段；逐帧解码，记录 SHA-256 | 本项目脚本；时间使用原视频帧号/FPS，结束帧不包含在区间内 |
| 镜头切分 | 灰度变化、HSV 直方图变化、局部变化基准和峰值抑制；补充渐变转场 | 本次 AI 辅助编写的启发式实现，参考自适应切镜思想，未复制或调用 PySceneDetect |
| 人物候选 | 每镜头抽取 15%/50%/85% 三帧，检测人体框与关键点 | 使用项目已有的 Ultralytics YOLO11n-pose 预训练权重；未重新训练 |
| 身份关联 | 用此前自动踢球者/守门员候选建立球衣参照；HSV 环形色相平滑、亮度边缘化 | 本项目启发式；红色色相首尾相接；颜色只提供身份候选，同队球员仍可能混淆 |
| 守门员参照修正 | 将上游髋部中心匹配回其同帧原始人物框 | 没有把髋部中心误当成人物框中心，也没有凭人工标签补齐缺失项；38 段有参照，Kramaric 的声明缓存缺失 |
| 机位判断 | 人物相对大小/前后位置、球门几何、边缘近网线索、跨抽样投票 | 本项目规则；白色掩膜、Canny/Hough 来自 OpenCV API。球门模块默认参数保持原样，机位实验单独允许更宽的球门 |
| 斜机位方向 | 标准球门四角 + 多组假设内参，比较 PnP 方向的一致性 | OpenCV `solvePnP`；内参、球门角点均未标定，因此只输出方向候选，不宣称精确 45° |
| 慢动作尝试 | 8 Hz 抽样，以六个归一化关节角比较同一次动作，检查顺序和相对时间拉长 | 本次编写的实验算法；2D 角度不具有完整视角不变性，身体裁切常导致弃权；正常速度参照也未人工验证 |
| 导出与复核 | 输出 JSON/CSV、三帧图、H.264 视频及本地页面 | 所有自动结果都有 `review_required=true`；人工复核只能由使用者主动保存，单独导出草稿 |

方法参考：[PySceneDetect detector 文档](https://www.scenedetect.com/docs/latest/api/detectors.html)、
[OpenCV PnP 文档](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)。
该模块不是 SoccerNet 代码移植，没有读取比赛进球结果来推断镜头类别；预训练模型、
OpenCV API、项目已有模块和本次新写的规则应在答辩时分别说明。

### 文件与使用

- `camera_timings_candidates.csv`：每个候选镜头的起止秒、帧号、机位候选、速度状态、作业类别候选。
- `camera_timings_candidates.json`：完整证据和每段视频四类目标的 `detected_candidate` / `not_detected` 状态。
- `summary.json`：本次统计；`artifact_audit.json`：原视频哈希、时间轴、视频解码与针对性测试。
- `scan/`、`features/`：原始切镜/检测缓存；`references_v2/`、`context_v2/`：修正参照及机位几何。
- `action_v3/`：当前动作对齐结果；旧 `action/`、`action_v2/` 为保留的实验，不是最终结果。
- `review/index.html`：中文复核页面；点击时间段即可跳转；支持前/后一帧和复核草稿导出。

本机打开：

```powershell
& 'C:\Users\30605\Virtual_Environment\myenv\Scripts\python.exe' -X utf8 scripts/serve_penalty_phase_review.py --directory outputs/camera_timing_2026-10-10/release/review --port 8884
```

然后打开 `http://127.0.0.1:8884/index.html`。服务仅监听本机，只读，无上传接口；
只提供复核目录，不公开项目其他文件。浏览器人工草稿不会更新模型结果，也没有被当作已验证真值。

从头复现请使用**新的输出目录**，避免把旧缓存与新规则混用：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/run_camera_timing_analysis.py --stage all --output-dir outputs/my_camera_experiment
python scripts/prepare_camera_review_media.py --output-dir outputs/my_camera_experiment
python scripts/audit_camera_timings.py --output-dir outputs/my_camera_experiment
```

运行依赖记录在 `requirements-camera.txt`。本机使用上述 `myenv` Python，推理时仅将
项目 `.venv/Lib/site-packages` **追加**到搜索路径以访问已有 Ultralytics，避免覆盖当前
解释器的 NumPy/Torch。FFmpeg 使用已存在的
`outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`。
上游角色/接触帧/姿态缓存和本地权重是前提；它是独立运行的第二问脚本，但并非不依赖
其他项目文件的通用单文件程序。`--clips` 可限定清单中的一个或多个 clip ID。

阶段 `scan`、`features`、`references`、`context`、`action`、`export` 可以分开运行。
`--resume` 只复用匹配的缓存，代码或输入变化时可能拒绝复用；不要靠改哈希绕过检查。
当前 release 的旧原始检测缓存有明确的后续参照/规则修正，更新发布文件只需
`--stage export`；不要用新版 `all --resume` 强行覆盖该历史实验。

### 目前未完成的部分

197 个镜头候选覆盖 39 段视频、21,450 帧，不等于 197 个镜头都分类成功。
`unknown` 同时包括证据不足、人物/教练特写、非目标镜头等，不能直接当错误率。
没有人工真值，因此不报告分类准确率、召回率或边界误差的实测分数。

AI 抽样查看发现明显的漏判例子：Cuadrado 的背后回放在球门被裁切后丢失几何支持；
Kane 的近景回放中人物身体被画面边缘裁切，颜色及完整关节角不稳定；Smolov 的门后
镜头中远处罚球者过小，需依赖近网和守门员线索。这是 **AI 的失败分析，不是用户人工复核**。
重复帧、球不动、低 FPS 都不能单独证明慢动作。即使得到时间拉长候选，也须核实动作
身份、参照速度和跨机位对齐；当前两类慢动作不能宣称已完成。

后续优先项是补充镜头内的抽样/跟踪、处理不完整身体的动作特征、检查完整比赛中的回放
范围，并建立单独的真实复核集。不能通过将所有未知强制归入四类来提高表面完成率。

本轮针对机位、球门几何及相邻球门实验的 35 项测试通过；完整测试发现曾因当前环境
没有 `pytest` 而出现一个模块导入错误，不能宣称全项目测试全绿。`browser_qa.json`
记录实际页面、MP4 Range、切换视频、时间跳转和区间末帧停止检查；
`assistant_visual_review.json` 仅记录 AI 观察到的样例问题，没有创建人工标签。
