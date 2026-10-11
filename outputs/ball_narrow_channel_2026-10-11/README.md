# 球：窄通道关联实验

39 个主镜头片段全部重新关联，每段执行两次，复用此前神经检测和原像素运动候选，没有重新运行神经网络。此前自动静止种子保持一致，没有输入人工球坐标或进球标签。

通道沿更早实测点的运动方向向前延伸，允许 3 像素向后测量误差。通道半宽至少 18 像素，按此前候选尺寸和漏帧数少量增加，最多 32 像素。新目标不能扩大自己的接受范围。青色通道在关联当前帧之前建立，预测不是实测球点。

通道外至少 55 度的大转向进入另一条假设，仍需之前球位置附近的自动碰撞几何线索和至少 3 个连续观测支持。几何接近不等于发生接触，不能因此判定没进。缓慢弯曲可通过逐帧更新方向继续跟踪。

运动差分可能包含已经离开的旧位置。新版本检查运动候选框在当前原画面是否有亮色、低饱和度像素；缺乏支持只降低得分。严格掩膜在复测中提前截断了一些轨迹，因此最终关闭该硬门槛。全部原始候选仍保留，神经检测候选不被掩膜删除。白线、球鞋、衣物仍可能通过，阴影中的球也可能缺乏支持。

主要关注触球后射门和球门附近约 0.2 秒的观察；窗口上限为自动触球后 0.85 秒。球门后的球网运动不用于判没进。当前结果仍为 `unknown`，二维通道无法确认门线深度、全球越线或最终停球位置。

绿色是窄通道和当前像素支持的新选择，蓝色是相同原始候选和短窗口的原关联对照，橙色是短观察。本轮变化包含通道和像素支持两项，不能全部归因于通道。旧完整窗口及旧报告保留，未替换生产跟踪入口 `src/ball_continuity.py`。

`tests.json` 是合成规则和回归检查；`validation.json` 记录重复运行、候选成员、源视频哈希及视频解码；`visual_review.json` 是 AI 原像素抽样核对。三者均不是人工逐帧真值或准确率。报告同时列出真实球点遗漏、同方向残影和其他残留误跟。

复现到一个不存在的新目录：

```powershell
& 'C:/Users/30605/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 -m scripts.run_ball_direction_corridor_review --output-dir outputs/ball_narrow_channel_replay
& 'C:/Users/30605/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 -m scripts.build_ball_corridor_evidence --directory outputs/ball_narrow_channel_replay
```

依赖 `outputs/ball_mask_review_ready_2026-10-11` 的原候选缓存和自动种子、`outputs/ball_detection_all_2026-10-11/detections`、原始视频及 FFmpeg。`protocol.json` 和 `source_snapshot` 固定本次参数与代码。抽样视觉评语不由算法重跑生成。
