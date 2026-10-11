# 守门员反弹关联定向核对

用户指出 Bacca 被扑出后没有继续跟踪。本轮保持来球前缀，使用自动守门员几何、接触时空一致性、至少三次独立观测、当前白球像素与短反弹窗口重新关联。导出保留旧版蓝圈、真实候选绿圈与独立橙色预测；换向待确认时暂停旧通道，确认后使用先前的反弹观测建立新通道。

本轮为 Bacca、Henderson、Kane 三段的定向实验。Bacca 新关联的六个球影经过 AI 原像素核对；Henderson 的新球影和 Kane 假分支拒绝也经过 AI 核对。未作全部39段准确率验收，未输入人工球坐标或结果标签。此实验不自动判断是否进球。

查看 report.html。复现关联与渲染（指定一个新的输出目录）：

    python -X utf8 -m scripts.review_ball_keeper_rebound --fresh-dir outputs/ball_keeper_rebound_inputs_2026-10-11 --output-dir outputs/NEW_DIRECTORY --clips col_eng_p09_bacca_col7 col_eng_p06_henderson_eng8 col_eng_p02_kane_eng9

复现新的局部神经检测需要项目 .venv 的 Python 3.14 与 torch/ultralytics。缓存来自 scripts/cache_ball_keeper_contact.py（目标球门ROI、classes=0/32、imgsz=960、conf=.01），原文件与模型哈希见缓存；依赖环境、当前像素筛选和输入发生变化时须重新验证。运行规则检查：tests.test_ball_keeper_rebound 以及 validation.json 中记录的回归日志。
