# 全部 39 条球识别复测

本轮从原视频重新推理，旧球轨迹没有作为输入，模型权重和当前跟踪参数不变。
39 条标准点球的主射门窗口全部完成，共 2,920 帧；处理错误 0，39 个慢速回放均完整解码。
窗口采用现有自动主镜头/触球提议，从触球前 0.4 秒至后 2.3 秒，并受主镜头结束限制。
这不是对原始视频中所有机位、回放或全部帧的重新推理。Aspas 自动窗口仅 27 帧。

## 当前结论

- AI 逐条原像素核对：39 条各至少有一次神经检测命中真球，主要在触球前。
- 自动静止起点代表帧：35 条是真球，4 条选错。选错的是 Ramos、Cheryshev、Aspas 和 Vida。
  这 4 条的原始神经检测中同样有真球，但当前起点选择没有选中。
- 含运动关联的抽样：30 条确认跟错目标；6 条部分跟上但没有全帧验收；
  1 条接触附近身份不清；2 条没有建立真实飞行段。
- Iniesta 第 54 帧仍有真实飞行球检测，第 67 帧却关联到网与广告，复现此前指出的问题。
  `ball_continuity_preserved_2026-10-11` 的旧轨迹保留结果没有被改写。

39/39 是“每条至少一次可见真球检测”的 AI 核对覆盖，不是准确率。
30 条是抽样发现的失败片段，不是全帧误检率。没有新增用户人工真值，
没有计算准确率、召回率、HOTA 或球心误差；预测点和没有足球身份的亮斑不计为真球神经检测。
本轮只复测，没有修改模型或连续性跟踪代码，也没有训练或自动判进球。

## 查看证据

- `report.html`：中文筛选页面，39 个慢速回放、233 个不同的片段内抽样帧，以及原像素放大。
- `visual_review.json` / `visual_review.csv`：AI 抽样结论、帧号和每条至少一个真球检测 ID。
- `validation.json`：原视频/权重/代码/输出视频哈希和范围检查。
- `protocol.json` / `source_snapshot/`：本轮冻结的主要代码与参数。
- `detections/`：39 条新推理原始候选；`results/`：纯检测与含运动两套关联及单独预测。
- `seed_evidence.json` / `extra_evidence.json`：视觉证据来源。额外点选只用于检查，未输入推理或关联。

视觉核对范围是每条自动种子代表帧、每条六帧对照图及必要附加局部图，
不是人工逐帧真值验收。回放右侧优先显示绿色关联，其次蓝色关联/静止起点/预测，
应结合主画面颜色区分。橙色加号仅预测。

浏览器工具禁止打开本地 `file://` 地址；页面交互和浏览器播放没有自动验收。
所有回放已在生成阶段逐帧解码验证，图片和本地引用均作文件检查。
在本机浏览器手动打开 `report.html` 即可查看。

## 复现

使用能导入 OpenCV、NumPy、Torch、Ultralytics、PyYAML 的项目 Python 重新推理；
模型来自本地 `models/weights/yolo11n_object_detection.pt`，FFmpeg 使用已有项目工具。
必须使用新的输出目录，防止把旧候选与新推理混用：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python -X utf8 scripts/retest_ball_detection_all.py --output-dir outputs/ball_detection_retest_new
```

`--resume` 只用于同一输出目录中代码/权重哈希一致的续跑。
视觉结论是本轮 AI 检查记录，不会自动转移到新的推理结果。
原像素证据与最终报告脚本使用 Pillow 和本地 FFmpeg，可以采用另外的可用解释器：

```powershell
python -X utf8 scripts/build_ball_detection_seed_evidence.py --result-dir outputs/ball_detection_all_2026-10-11
python -X utf8 scripts/build_ball_detection_review_extras.py --result-dir outputs/ball_detection_all_2026-10-11
python -X utf8 scripts/finalize_ball_detection_retest.py --result-dir outputs/ball_detection_all_2026-10-11
```

最终报告依赖本轮已经完成的 `visual_review.json`，不会自行生成新的人类标签。
