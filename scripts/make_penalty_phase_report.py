"""Render a source-linked morning review report from the actual run JSON."""
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/phase_optimization_2026-10-09"
STYLE = {
    "pace_increase_image_plane": "图像平面提速",
    "curved_approach_image_plane": "弧线助跑候选",
    "curved_preparation_image_plane": "准备期弯曲轨迹候选",
    "lateral_adjustment_relative_to_approach_image_plane": "横移调整候选",
    "loop_like_preparation_image_plane_candidate": "绕圈状投影候选",
    "pause_or_stutter_image_plane_candidate": "暂停/节奏变化候选",
    "dense_low_progress_movement_image_plane_candidate": "密集、低推进量动作候选",
    "multiple_small_ankle_excursion_candidates": "小幅踝部运动候选",
}
CONTACT = {"supported": "较强证据", "tentative": "暂定", "abstain": "弃权"}


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    summary = load(RUN / "classified/summary.json")
    comparison = load(RUN / "model_comparison.json")
    validation = load(RUN / "validation.json")
    clips = [load(RUN / "classified" / f"{row['clip_id']}.json") for row in summary["rows"]]
    tags = Counter(tag for r in clips for tag in {t["tag"] for t in r["phase_result"]["style_tags"]})
    rows = []
    for r in clips:
        central = r["count_sensitivity"].get("central", {})
        name = r["clip_id"].split("_")[3]
        count = central.get("approach_candidates_including_precontact_strike_window")
        prep = central.get("preparation_candidates")
        styles = "、".join(STYLE.get(t["tag"], t["tag"]) for t in r["phase_result"]["style_tags"]) or "未取得特定风格证据"
        if r["phase_result"]["status"] == "abstained":
            styles = "不强行分类：触球/人物连续性证据不足"
        early, late = central.get("approach_count_with_early_contact_boundary"), central.get("approach_count_with_late_contact_boundary")
        values = [name, r["clip_id"], CONTACT[r["contact"]["status"]], str(prep) if prep is not None else "—",
                  str(count) if count is not None else "—", f"{early}–{late}" if early is not None else "—", styles]
        rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    tag_list = "".join(f"<li>{html.escape(STYLE.get(tag, tag))}：{number} 段</li>" for tag, number in tags.items())
    abstained = "、".join(r["clip_id"] for r in clips if r["phase_result"]["status"] == "abstained")
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vision II · 动作分类实验验收报告</title><style>
body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#213b36;background:#f4f7f4;margin:0}}main{{max-width:1160px;margin:auto;padding:36px 24px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:34px}}a{{color:#156756}}.note{{padding:16px;background:#fff1d8;border-left:4px solid #c78b34}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #d9e5df;text-align:left;vertical-align:top}}td:nth-child(2){{overflow-wrap:anywhere;min-width:160px}}th{{background:#e0ede5}}code{{overflow-wrap:anywhere}}li{{margin:7px 0}}@media print{{body{{background:white}}main{{padding:10px}}}}
</style><main><p>IND5003 / VISION II / 2026-10-09</p><h1>把“数一个总步数”改为“先分动作阶段，再复核候选”</h1>
<p><a href="index.html">打开完整视频复核台 →</a>　所有材料均在原项目文件夹内；原视频和历史主表没有覆盖。</p>
<p class="note">这是已实现、已运行的候选分类实验，不是经过人工真值验证的准确率提升。39 段全部处理，{summary['phase_status'].get('provisional',0)} 段生成暂定阶段，{summary['phase_status'].get('abstained',0)} 段不强行分类。零条独立逐帧真值；唯一用户意见是 Cuadrado 大约5步、该例4–6步可接受。</p>
<h2>1 · 怎么分类</h2><p>时间阶段：准备/调整 → 起步窗口 → 接近球/助跑 → 踢球附近 → 随动恢复。横移、弧线、绕圈状、提速、停顿、低推进量密集动作是可以并存的描述，不是互斥的人物类型。“起步窗口”按规则定义，并不自动代表测得了加速度。</p>
<p>当前真实运行得到的描述如下。同一段可以有多个标签；没有标签不是“没有这种动作”。没有足够的横移或完整绕圈证据时，保留未判定，不为了凑类别强行分配。</p><ul>{tag_list}</ul>
<h2>2 · 每一步做了什么</h2><ol>
<li>复用原始逐帧人物框和COCO-17姿态缓存，重新审核人物角色。AI看过39张旧QA图，发现11条旧选中轨迹明显落在门将或裁判；这不是人工标注。</li>
<li>不再把踝部运动最大值直接当触球。用YOLO球候选、相对静止位置、所有人物脚踝的接近与球离开证据，估计罚球者和触球窗口。最终{summary['contact_status'].get('supported',0)}条较强证据、{summary['contact_status'].get('tentative',0)}条暂定、{summary['contact_status'].get('abstain',0)}条弃权。这是证据分级，不是正确率。</li>
<li>Kovačić旧镜头[0,39)仍是开场人物画面；自动搜索后续镜头，选[40,133)，独立触球候选77帧。禁止沿用该镜头旧计数器仍给出的116帧。</li>
<li>在人物框外采样背景角点，Lucas–Kanade光流、正反向一致性和RANSAC仿射变换补偿镜头运动。任何失效变换都会中断参考坐标链，不再静默假定镜头没动。</li>
<li>髋部和脚踝置信度过滤、短缺帧插值、长缺帧保留unknown。髋部速度、运动段、相对方向、0.4秒窗口的累计前进量共同限定起步；早期身体晃动不再只凭瞬时速度进入助跑。</li>
<li>脚踝相对位置信号归一化，PCA提取主要运动轴，SciPy峰谷产生事件候选。三档显著度0.025/0.04/0.06固定展示敏感性，未用用户5步意见拟合参数。候选不是已确认的脚掌触地。</li>
<li>将候选归入准备、起步、助跑、触球窗口等阶段。触球窗口可能包含支撑脚接近或摆腿，单独显示边界敏感性，避免强行多算/少算一步。</li>
<li>以相邻候选之间的稳定化髋部推进量和时间间隔，描述“密集、低推进量”片段。它不是米制步长或真实步频，也不据此删去小步。</li>
<li>额外跑YOLO11m人体裁剪对照。有效髋踝样本只从2661变2662，未证明准确率提高；默认保留原缓存姿态，增强版只供比较。</li></ol>
<h2>3 · 消融和验证</h2><p>真正可比较的增强姿态有{comparison['actual_refined_comparisons']}条；{comparison['count_changed']}条名义助跑候选数变化，{comparison['count_difference_at_least_two']}条相差至少2，{comparison['onset_difference_more_than_four_frames']}条起点相差超过4帧，{comparison['style_tag_set_changed']}条描述标签集合不同。选中模型不能只因为它更符合预期。</p>
<p>{validation['unit_regression_tests']['tests_run']}项选定单元/回归检查通过情况：{validation['unit_regression_tests']['passed']}。两版共78条结果检查绝对帧号、分段连续性、候选数一致性、源码哈希和QA图片解码。39个H.264播放副本均完整解码并核对帧数、帧率、尺寸；这些都是工程检查，不是实际视频准确率。</p>
<h2>4 · 早上优先看哪些</h2><ol><li>Cuadrado：原姿态名义助跑候选仍为5，增强版可能不同；先盲看，再看候选和起点。</li><li>Uribe：准备期与助跑期分开，重点判断准备期峰谷到底是调整步还是姿态抖动。</li><li>Rashford / Pivarić：候选较密，不能仅因数字大判错；核对是否存在真实小步、交替摆腿或重复检测。</li><li>ESP–RUS Ignashevich / RUS–CRO Modrić / Kovačić：优先核对黄框是否一直跟随罚球者及触球窗口。</li><li>Ramos / Schøne / Aspas：两种姿态的计数分歧较明显，适合逐帧确定哪些事件是真的。</li></ol>
<h2>5 · 全部球员的候选摘要</h2><p>只描述本次罚球，不概括球员长期固定策略。准备期数字也只是候选事件，不是已确认的准备步。边界范围不是统计置信区间。</p><div class="table"><table><thead><tr><th>球员</th><th>视频编号</th><th>触球证据</th><th>准备候选</th><th>助跑候选</th><th>触球边界敏感性</th><th>动作描述</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>6 · 尚未解决</h2><p>未强行分类：{html.escape(abstained)}。</p><p>Kramarić关联到很短的触球轨迹，不能代表完整助跑；其余主要缺少足够可信的球/脚接近与离开证据。画面肉眼清晰并不代表检测器已识别正确。这些是具体算法限制，不归咎于“数据少”。</p>
<p>仍需人工确认每条视频的计数定义、准备/正式助跑分界、真实落脚帧和触球帧，才能评估MAE、阶段准确率及模型优劣。未经球场标定不能报告真实米制速度、冲刺或地面圆形路径；单次视频也不能证明门将依赖型策略。</p>
<h2>7 · 来源与可复现性</h2><p>检测权重来自Ultralytics；角点/光流/仿射估计调用OpenCV；信号峰谷调用SciPy。本次阶段划分、阈值、拒判和证据组合是本项目新编写的启发式代码，未冒充课程/SoccerNet现成算法。</p>
<ul><li><a href="https://docs.ultralytics.com/tasks/pose/">Ultralytics pose官方文档</a></li><li><a href="https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html">OpenCV光流官方文档</a></li><li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html">SciPy find_peaks官方文档</a></li><li><a href="https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2022.818556/full">姿态估计与点球观察分析研究</a>：仅为区分可观察动作与策略推断的概念参考，不复制其OpenPose流程或借用其准确率。</li></ul>
<p>完整复现命令和方法边界：项目 <code>docs/PENALTY_PHASE_EXPERIMENT_2026-10-09.md</code>。源数据、代码哈希与中间结果保存在每条JSON；无上传、无覆盖历史主结果。</p>
<small>生成时间 UTC：{datetime.now(timezone.utc).isoformat(timespec='seconds')} · AI辅助实验 · human_verified=false</small></main></html>'''
    destination = RUN / "review/report.html"
    destination.write_text(report, encoding="utf8")
    print(destination)


if __name__ == "__main__":
    main()
