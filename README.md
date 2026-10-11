# Vision II 点球视频分析

本项目分析四场世界杯点球大战的 **39 次点球**，提取罚球者、守门员、足球和进球事件的视觉证据，并定位视频中的镜头时间段。项目包含 Python 源码、预训练模型、检测缓存、实验结果、逐帧诊断、回放页面和 Notebook，供队友查看、复跑和继续改进。

**GitHub 已补齐分析用的项目文件。队友需要下载仓库和补充包、安装 Python 依赖，并放入已单独分享的 rawclip。** 球的缓存复跑已在另一个项目目录验证通过；全新电脑的依赖安装和所有历史实验从头运行尚未逐项验证，因此目前不能承诺一个命令重算全部历史工作。

## 项目具体做什么

| 分析对象 | 做什么 | 可以查看什么 |
|---|---|---|
| 罚球者 | 用人体姿态选择罚球者，估计触球帧、左右触球脚、助跑轨迹和步数；另有动作分阶段、末段碎步和助跑方式实验 | 步数候选、关键点轨迹、触球证据和复核页面 |
| 守门员 | 区分门将与裁判，分析罚球前移动、扑救方向；用 SAM2 掩膜补充跟踪，分析首次扑救阶段和图像高度 | 人物选择证据、分割回放、方向与高度候选 |
| 足球 | 组合 YOLO 足球检测、颜色/运动掩膜、静止种子、连续运动预测和窄通道关联，重点处理射门及门前短时反弹 | 原始检测、观察与预测的区别、轨迹、球门范围和慢放 |
| 是否进球 | 根据球门范围、连续到达证据和有支持的反弹给出 `goal_likely`、`no_goal_likely` 或 `unknown` | 自动规则理由，以及单独保存的 AI 画面判断 |
| 机位和时间段 | 切分镜头，结合人物、球门几何和动作对齐，生成课程机位类别与慢动作候选 | 镜头起止帧/秒、机位候选、慢动作证据和复核页面 |
| 数据汇总 | 按唯一 `clip_id` 汇总特征，通过 Notebook 和统计图探索关系 | 39 次点球的特征表、分布和诊断图 |

正式样本清单是 [点球切片配置](configs/penalty_slicing_manifest.yaml)：哥伦比亚—英格兰 10 次、克罗地亚—丹麦 10 次、西班牙—俄罗斯 9 次、俄罗斯—克罗地亚 10 次。文件夹中的旧命名或重复切片不增加正式样本数。

## 从 GitHub 下载完整项目

### 1. 准备工具

- 使用 Windows 64 位，安装 Python、Git for Windows 和 Git LFS。
- 仓库是公开仓库，无需协作者邀请或登录即可查看和下载。
- 主流程已有运行记录的环境是 Windows CPU / Python 3.14.7；依赖记录见 [requirements-pycharm.txt](requirements-pycharm.txt)。该文件尚未经过全新环境安装验证。
- 项目文件、LFS 缓存和还原文件建议预留约 **13 GB** 空间；Python 环境及原视频还需要额外空间。

### 2. 克隆并下载补充包

在 PowerShell 中进入准备存放工程的目录，执行：

```powershell
git lfs install
git clone https://github.com/RX422040/vision-ii-penalty-analysis.git
cd vision-ii-penalty-analysis
git lfs pull
python -X utf8 restore_project_files.py
```

两个补充包位于 `project_files/`，合计约 **3.58 GB**。还原脚本会校验压缩包和文件的 SHA-256，再将 **26,892 个文件** 放回模型、历史输出、Notebook 归档和 SAM2 辅助目录。它负责还原文件；安装依赖和运行分析使用后续命令。

成功时会打印 `"all_file_hashes_match": true`。需要再次核对时执行：

```powershell
python -X utf8 restore_project_files.py --check
```

相同文件会复用。已有文件与补充包不同会停止，避免覆盖本地实验；保留本地改动后，可在新克隆目录重新还原。

### 3. Git LFS 下载失败时

打开 [项目 Release 下载页](https://github.com/RX422040/vision-ii-penalty-analysis/releases/latest)，下载这两个附件：

- `visionii-support-01.zip`
- `visionii-support-02.zip`

将它们放入工程的 `project_files/`，替换同名的小型 LFS 指针文件，再运行 `restore_project_files.py`。

GitHub 的 **Code → Download ZIP** 可能只提供 LFS 指针；不要把几十或几百字节的指针文件当作完整补充包。Release 附件提供相同内容，仍会由还原脚本校验。

### 4. 放入原视频

把已单独分享的 rawclip 解压到：

```text
data/
  raw_clips/
    col_eng_p01_falcao_col9.mp4
    col_eng_p02_kane_eng9.mp4
    ...
```

视频应直接位于 `data/raw_clips/`，不要多套一层同名文件夹。保留原文件名和原文件字节，不要重新编码后覆盖；缓存和保存的 AI 画面判断会核对来源视频。

**原视频没有放进这两个补充包。** 常规特征分析使用 rawclip 即可；若要重新切片，还需要原始完整比赛视频，放在 `data/full_videos/`，具体文件名见切片配置。已分享的旧命名切片也可保留，以便使用历史脚本。

## 配置 Python 和 PyCharm

在工程根目录创建自己的虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-pycharm.txt
```

在 PyCharm 中打开仓库根目录，将项目解释器设置为本工程的 `.venv/Scripts/python.exe`。不要使用原作者电脑的解释器路径。

新建一个 **Python Run Configuration**：

| 配置项 | 填什么 |
|---|---|
| Script path | 工程根目录下的 `run_project.py` |
| Working directory | 工程根目录 |
| Python interpreter | 本工程 `.venv/Scripts/python.exe` |
| Parameters | 先填 `check`；之后按下面的模式切换 |

下面的 `python` 命令应在选用上述解释器的 PyCharm Terminal 中运行；也可以把 `python` 换成 `.\.venv\Scripts\python.exe`。

主环境优先使用 `requirements-pycharm.txt`。其它 `requirements-*.txt` 保留的是对应实验的依赖记录，不要全部叠加安装。球衣 OCR 使用单独的 Python 3.12 环境和 [requirements-jersey.txt](requirements-jersey.txt)；SAM2 源码和辅助模块随仓库及补充包提供。

## 推荐先完成这三步

```powershell
python -X utf8 run_project.py check
python -X utf8 run_project.py report --open
python -X utf8 run_project.py replay-ball --output-dir outputs/my_ball_run --open
```

1. `check` 检查球流程所需的导入、39 段原视频、检测缓存、四个权重文件和 Windows FFmpeg。输出 `READY` 表示这些前提齐备，尚不代表其它模块全部验收通过。
2. `report --open` 打开已经保存的 39 次球事件报告，不重新检测。
3. `replay-ball` 用已有检测缓存重新执行关联、反弹和进球规则，生成新报告与慢放。这个模式不调用新的 YOLO 推理。

输出目录必须在本工程的 `outputs/` 下，且不能已存在。重复运行时换一个目录名，例如 `outputs/my_ball_run_2`。

## 各个运行模式怎么用

| 模式 | 具体用途 | 主要输出 / 说明 |
|---|---|---|
| `check` | 检查球流程运行前提 | 终端中的缺失项或 `READY` |
| `report --open` | 查看已保存的球事件报告 | 浏览器打开现有报告 |
| `test-ball` | 运行球规则、连续性、反弹及路径测试 | 测试日志；通过不等于真实检测准确 |
| `replay-ball` | 利用缓存重新计算全部 39 次球事件 | 新目录中的 `report.html`、`outcomes.csv`、诊断 JSON 和视频 |
| `refresh-ball` | 对 39 段的门前短窗口重新进行 YOLO 足球检测，再执行球事件流程 | 同样的报告，另外生成检测缓存目录；仍使用已有镜头、球门、种子和入射轨迹证据 |
| `steps` | 重新运行基础姿态与助跑步数流程 | `penalty_step_counts.csv`、`step_diagnostics.json`；不是全部步数优化实验的一键入口 |
| `camera` | 重新切镜、检测人物、分析机位和动作对齐 | 镜头时间段 CSV/JSON、缓存、复核页面；页面视频准备另见下文 |
| `goalkeeper` | 运行全部 39 段的首次扑救阶段和图像高度实验 | `summary.json`、`report.html`、分割/姿态证据和回放；依赖已有角色和方向结果 |
| `notebook` | 启动 JupyterLab 查看、编辑 Notebook | 打开 `notebooks/`，不自动执行所有单元格 |

可直接使用的命令：

```powershell
python -X utf8 run_project.py test-ball
python -X utf8 run_project.py refresh-ball --output-dir outputs/my_ball_fresh_run --open
python -X utf8 run_project.py steps --output-dir outputs/my_steps_run
python -X utf8 run_project.py camera --output-dir outputs/my_camera_run
python -X utf8 run_project.py goalkeeper --output-dir outputs/my_keeper_run
python -X utf8 run_project.py notebook
```

`refresh-ball`、姿态、机位和 SAM2 分割需要模型推理，CPU 耗时明显长于打开报告或缓存复跑。`refresh-ball` 会另建 `outputs/my_ball_fresh_run_fresh/`；主目录及这个缓存目录都必须使用新名称。它重新检测门前窗口，尚不是从原视频重建全部上游证据的流程。

### 机位分析后生成可播放页面

`camera` 导出镜头表和页面后，准备页面使用的媒体，再检查导出：

```powershell
python -X utf8 scripts/prepare_camera_review_media.py --output-dir outputs/my_camera_run
python -X utf8 scripts/audit_camera_timings.py --output-dir outputs/my_camera_run
python -X utf8 scripts/serve_penalty_phase_review.py --directory outputs/my_camera_run/review --port 8884
```

保持最后一个命令运行，在浏览器打开 [本机机位复核页面](http://127.0.0.1:8884/index.html)。结束查看时在终端按 `Ctrl+C`。

`camera_timings_candidates.csv` / `.json` 记录镜头起止帧、起止秒、机位与速度候选。时间按原片帧号/FPS计算，结束帧不包含在区间内。四类课程机位与慢动作仍存在漏判和 `unknown`。

## 已做过的结果在哪里看

还原后，以下链接对应本地文件。GitHub 网页可以查看文本、CSV 和 JSON；HTML 回放应下载后在本地浏览器打开，GitHub 文件预览不会直接播放项目页面。

| 要看什么 | 入口 |
|---|---|
| 球跟踪与是否进球总报告 | [球事件报告](outputs/ball_goal_all_events_v2_2026-10-11/report.html)、[事件表](outputs/ball_goal_all_events_v2_2026-10-11/outcomes.csv)、[复核摘要](outputs/ball_goal_all_events_v2_2026-10-11/review_summary.json) |
| 守门员 SAM2 方向候选 | [方向报告](outputs/goalkeeper_sam2_update_2026-10-11/results_v4_release/report.html)、[39 条结果](outputs/goalkeeper_sam2_update_2026-10-11/results_v4_release/goalkeeper_results_39.json) |
| 守门员首次扑救阶段和高度 | [高度报告](outputs/goalkeeper_height_phase_2026-10-11/release_v2/report.html)、[证据汇总](outputs/goalkeeper_height_phase_2026-10-11/release_v2/summary.json) |
| 助跑方式与末段调整 | [步数页面](outputs/phase_optimization_2026-10-09/review/steps/index.html)、[末段调整页面](outputs/phase_optimization_2026-10-09/review/terminal/index.html) |
| 机位时间段 | [机位复核页面](outputs/camera_timing_2026-10-10/release/review/index.html)、[时间段表](outputs/camera_timing_2026-10-10/release/camera_timings_candidates.csv) |
| 不完整人体的机位补充实验 | [机位补充报告](outputs/camera_partial_alignment_2026-10-11/report.html) |
| 已保存的综合特征与诊断 | [39 条汇总](outputs/optimization_evening_2026-10-09/integration/canonical_results_39.json)、[统计记录](outputs/optimization_evening_2026-10-09/integration/metrics.json)、[诊断图](outputs/optimization_evening_2026-10-09/integration/actual39_diagnostics.png) |
| 总体探索和代码流程 | [主 Notebook](notebooks/vision2_penalty_analysis.ipynb)、[步数 Notebook](notebooks/vision2_step_style_review.ipynb)、[门将高度 Notebook](notebooks/goalkeeper_height_phase_review.ipynb) |

综合特征表是保存的一版实验汇总；球事件和门将后续结果在各自目录中，尚没有一个命令将所有改进自动合并成新的最终特征表。`data/processed/penalties_features.csv` 的旧 49 行示例表不作为正式 39 次点球的实证结果。

## 历史实验和进一步分析

仓库保留各轮源码和输出，以便比较方法。除了主入口外，下面的脚本用于对应实验；它们有各自的上游缓存和参数，并非互相替代的一键全流程。

| 想继续做什么 | 对应脚本 |
|---|---|
| 修复第一现场镜头和基础步数 | [rerun_step_shot_repair.py](scripts/rerun_step_shot_repair.py) |
| 助跑、触球、跟随动作分阶段 | [run_penalty_phase_experiment.py](scripts/run_penalty_phase_experiment.py) |
| 区分助跑和末段碎步 | [run_terminal_footwork.py](scripts/run_terminal_footwork.py) |
| 按单次罚球方式解释步数 | [run_style_aware_steps.py](scripts/run_style_aware_steps.py) |
| 利用已有接触/姿态证据重算优化步数 | [run_evening_step_reanalysis.py](scripts/run_evening_step_reanalysis.py) |
| 检查左右触球脚和跟随动作 | [run_goal_right_foot_redetection.py](scripts/run_goal_right_foot_redetection.py)、[analyse_goal_right_followthrough.py](scripts/analyse_goal_right_followthrough.py) |
| 门将与裁判身份关联 | [run_goalkeeper_role_experiment.py](scripts/run_goalkeeper_role_experiment.py) |
| 姿态扑救方向、SAM2 补充方向 | [run_goalkeeper_direction_update.py](scripts/run_goalkeeper_direction_update.py)、[run_goalkeeper_sam2_update.py](scripts/run_goalkeeper_sam2_update.py) |
| 球的静止种子、掩膜关联、窄通道、反弹 | [scripts 目录](scripts/) 中的 `run_ball_*`、`review_ball_keeper_rebound.py`；当前球事件使用主入口复跑 |
| 球衣号码 OCR 和机位补充 | [run_jersey_ocr_worker.py](scripts/run_jersey_ocr_worker.py)、[run_jersey_upright_ocr.py](scripts/run_jersey_upright_ocr.py)、[run_camera_partial_pose_experiment.py](scripts/run_camera_partial_pose_experiment.py) |
| 检查已有综合实验统计 | [vision2_evening.py](scripts/vision2_evening.py) 的 `inspect` 模式 |

例如，查看已有汇总、不重新推理：

```powershell
python -X utf8 scripts/vision2_evening.py inspect
```

使用已有末段动作证据生成另一版助跑方式结果：

```powershell
python -X utf8 scripts/run_style_aware_steps.py --output-dir outputs/my_style_run
```

历史脚本和 Notebook 中仍有原电脑的绝对路径，以及用于校验的源码/输入哈希。换电脑时可能需要迁移路径并重新生成相关缓存；不要靠关闭哈希检查复用不匹配的证据。部分脚本默认写入既有实验目录，运行前检查参数，支持时显式指定新 `--output-dir`。未提供独立输出参数的汇总或重分析脚本应在单独的工作副本中运行。

详细方法见 [研究思路与工作流程](docs/vision2_research_methodology_and_workflow.md)，课程对应和代码来源见 [来源说明](docs/vision2_code_provenance_course_soccernet_mapping.md)，课程原文见 [作业要求](docs/ind5003_vision_ii_project_requirements.pdf)。这些文档保留各自的分析背景；当前下载和主入口以本 README 为准。

## 怎样理解结果

- 球事件报告中的自动规则结果与 AI 画面判断是两个字段。保存的 AI 判断只用于展示，不是自动规则输入，也不是独立真人真值。
- 已保存球报告中，AI 画面判断为 26 次进球、13 次未进；自动规则为 20 次 `goal_likely`、6 次 `no_goal_likely`、13 次 `unknown`。事件 22、25、30 的两类判断存在冲突，需要复核。
- `unknown` / `review_required` 表示证据不足或需要核对，不应强制归类。进球后的球网运动、遮挡和杂物仍可能干扰；大角度变化也需要球身份、接触和多帧证据支持。
- 球轨迹、速度和门将高度主要是二维图像量或归一化代理量，不能直接写成真实米制速度、三维位置或已证明越过门线。
- 文件名含 `manual`、`confirmed` 或 `user_review` 并不自动证明完成真人标注。步数、机位和模型准确率需要另外建立独立复核集；测试通过、输出齐全和视频可播放不能作为准确率。
- 比赛结果标注在 [penalty_outcomes.csv](data/annotations/penalty_outcomes.csv)，来源和自动视觉判断应分别报告，不能把已知赛果当作检测成功。

## 已验证到哪一步

已有异目录克隆测试完成以下检查：

- 两个补充包的 26,892 个文件校验一致。
- 5 项还原脚本测试通过，72 项球规则与路径测试通过。
- 39 段球缓存复跑视频全部解码，共 1,433 帧，自动规则结果与保存版本一致。
- 机位模块的包导入已修正，18 项机位规则测试通过；一个样本的缓存参考生成与报告导出通过。这项检查未重新进行模型推理，也不代表 39 段机位分类已全部验证。

验证使用已存在的 Windows / Python 环境和原 rawclip，补充包通过本地相同文件验证，没有再重复下载占用 LFS 流量。**这不是全新依赖安装验证，也没有重新执行 YOLO 推理或全部历史模块。** 详细证据见 [克隆还原与球复跑验证](outputs/pycharm_export_validation_2026-10-11/git_clone_restoration_validation.json)。

## 常见问题

| 情况 | 怎么处理 |
|---|---|
| 无法克隆、Release 无法打开 | 核对仓库网址和网络连接；公开仓库无需协作者邀请 |
| 补充包只有很小的文本文件 | 执行 `git lfs pull`，或从 Release 下载两个真实 ZIP |
| 还原提示 checksum mismatch | 重新下载对应包，不要跳过校验 |
| 还原提示已有文件不同 | 保存本地改动，在新目录克隆还原；只有明确要替换时才使用 `--overwrite` |
| 缺少模型、FFmpeg 或历史缓存 | 先完成 `restore_project_files.py`，再执行 `check` |
| 找不到原视频 | 检查 `data/raw_clips/<clip_id>.mp4`，文件名和层级须对应清单 |
| AI 判断来源视频不匹配 | 放回原始字节的视频；换源后需要生成新的证据和判断 |
| 输出目录已存在 | 换一个新的 `outputs/` 子目录名 |
| pip 安装失败或缺少模块 | 确认解释器版本、当前虚拟环境和依赖文件；保存完整报错，当前没有全新环境安装通过的保证 |
| 历史脚本找不到原电脑路径 | 检查脚本及输入 JSON/Notebook 的路径，迁移后重新生成所需缓存 |
| 在 macOS / Linux 运行 | 包含的 FFmpeg 是 Windows 可执行文件，需要替换工具和相关路径；这两个平台未验证 |

## 目录说明

```text
run_project.py             PyCharm 主入口
restore_project_files.py   还原及校验补充包
requirements-pycharm.txt   主环境依赖记录
src/                       特征、跟踪、判定和几何算法
scripts/                   各模块及历史实验的命令行入口
tests/                     规则、路径与导出等测试
configs/                   配置和正式 39 次点球清单
data/raw_clips/            单独提供的原始点球切片
data/full_videos/          仅重新切片时需要的完整比赛视频
data/annotations/         来源标注、AI 候选和复核记录
data/processed/           基础流程保存的表格
models/weights/            还原后的 YOLO 与 SAM2 权重
outputs/                   检测缓存、实验结果、诊断、视频和页面
notebooks/                 探索与复核 Notebook
project_files/             Git LFS 补充 ZIP
docs/                      方法、来源、课程要求和文件清单
```
