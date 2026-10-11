"""Build the evidence-based Chinese progress audit, not the final course submission."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/vision2_completion_audit_2026-10-09.pdf"
pdfmetrics.registerFont(TTFont("Chinese", "C:/Windows/Fonts/simhei.ttf"))
FONT = "Chinese"
INK = colors.HexColor("#172334")
BLUE = colors.HexColor("#234667")
MUTED = colors.HexColor("#596777")
styles = {
    "title": ParagraphStyle("title", fontName=FONT, fontSize=23, leading=31, textColor=INK, spaceAfter=12, wordWrap="CJK"),
    "heading": ParagraphStyle("heading", fontName=FONT, fontSize=17, leading=24, textColor=INK, spaceAfter=13, wordWrap="CJK"),
    "sub": ParagraphStyle("sub", fontName=FONT, fontSize=12, leading=18, textColor=BLUE, spaceBefore=10, spaceAfter=6, wordWrap="CJK"),
    "body": ParagraphStyle("body", fontName=FONT, fontSize=10.5, leading=16.5, textColor=INK, spaceAfter=8, wordWrap="CJK"),
    "small": ParagraphStyle("small", fontName=FONT, fontSize=9, leading=13.3, textColor=MUTED, spaceAfter=6, wordWrap="CJK"),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=9.5, leading=14, textColor=INK, wordWrap="CJK"),
    "headcell": ParagraphStyle("headcell", fontName=FONT, fontSize=9.5, leading=14, textColor=colors.white, wordWrap="CJK"),
}
story = []


def p(text, style="body"):
    return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])


def add(text, style="body"):
    story.append(p(text, style))


def table(headers, rows, widths):
    data = [[p(t, "headcell") for t in headers]] + [[p(str(x), "cell") for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F4F7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), .6, BLUE),
        ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#CCD4DC")),
    ]))
    story.extend([t, Spacer(1, 10)])


def page(title):
    if story:
        story.append(PageBreak())
    add(title, "heading")


def footer(c, doc):
    c.setFont(FONT, 8.5)
    c.setFillColor(MUTED)
    c.drawString(44, 29, "Vision II 项目完成度审阅   2026年10月9日")
    c.drawRightString(A4[0] - 44, 29, f"{doc.page}")


def build():
    evidence = json.loads((ROOT / "outputs/completion_audit_2026-10-09/evidence.json").read_text(encoding="utf-8"))
    current = json.loads((ROOT / "outputs/completion_audit_2026-10-09/current_validation.json").read_text(encoding="utf-8"))
    assert current["passed"] and current["total_tests"] == 146
    style = evidence["json"]["outputs/style_aware_steps_2026-10-09/classified/summary.json"]
    assert style["clips"] == 39 and style["count_available"] == 31

    add("Vision II 项目完成度审阅报告", "title")
    add("截至 2026年10月9日  |  本地代码与结果重新核查", "small")
    add("结论", "sub")
    add("项目已形成能够展示和复查的多模块实验原型，但不能认定为 Vision II 全部完成，更不宜直接作为最终作业提交。真实视频的姿态、计步、守门员和球轨迹模块已有输出；俯视轨迹、自动 outcome、有效综合 EDA 和统一复现入口仍存在明确缺口。")
    add("本报告区分三件事：代码是否实现、多少条记录有候选结果、结果是否经过独立验证。以下百分比仅表示指定输出或规则筛选的覆盖率，不表示识别准确率，也不是老师的评分。")
    table(["模块", "当前可确认的完成度", "审阅判断"], [
        ["数据准备", "4场比赛；39个标准片段均存在", "已具备；原目录另有10个非标准名称片段"],
        ["罚球者", "旧左右脚候选35/39；新版步数31/39（79.5%）", "部分完成；真实俯视轨迹未完成"],
        ["守门员", "33/39有数值；18/39通过规则筛选（46.2%）", "部分完成；21条仍需复查"],
        ["足球", "39/39有轨迹；26/39通过规则筛选（66.7%）", "部分完成；终点只是最后观测点"],
        ["Outcome", "39/39已有赛果标签：26进、13未进", "标签整合完成；非视频自动识别"],
        ["机位与EDA", "有机位候选表和旧绘图脚本", "机位未验证；旧总表不能用于正式EDA"],
    ], [83, 226, 198])
    add("最重要的审阅发现", "sub")
    add("旧 penalties_features.csv 有49行，步数全部为1、踢球脚全部为right、结果全部为goal，多个运动量完全相同。对应旧流程使用手写示例轨迹；这些不是39次点球的真实联合测量。现存旧策略热图显示单一组合100%，不能作为研究结论。[E2]")
    add("新版步数结果在独立实验目录中，并未同步更新旧总表、守门员或球模块。主 notebook 仍残留“人工确认”文字，以及默认会被来源校验拒绝的重跑命令。现在应优先收拢真实流程，而非继续增加花样分类。[E4–E7]")
    add("范围：按原项目说明第4–7页审阅；重新读取本地代码、CSV、JSON、notebook和相关图像，并复跑选定测试。未重新训练或全量推理，未新增人工标签，未检查课程平台的上传状态。本报告是中文进度审阅，不是最终英文课程论文。", "small")

    page("一 逐项对照老师的要求")
    add("原要求包含9项核心特征、4类机位时间段及基础EDA。下表保留每项的实际含义，不用“有文件”代替完成。[E1]", "body")
    table(["要求", "代码与现有结果", "完成状态及尚缺证据"], [
        ["助跑俯视轨迹", "有单应性封装、39份球场线候选预览", "未完成真实标定；39份记录的标定完成标志均为False"],
        ["左右脚踢球", "旧CSV：21右、14左、4未知", "部分；需与新版选人和触球窗口重新关联，不能继承为已确认"],
        ["步数", "最新31条有候选，8条为空", "部分；峰谷不是已验证落脚；31/39为输出率"],
        ["守门员触球前移动", "33条有数值，6条unknown", "部分；依赖未验证触球帧与身份种子"],
        ["扑救方向", "33条有方向，6条unknown", "部分；image_left/right为画面左右"],
        ["扑救高度", "13 low、20 mid、6 unknown", "部分；人体尺度归一化的二维代理量，不是真实腾空高度"],
        ["球轨迹", "39条轨迹；26规则接受、13待查", "部分；尚无独立轨迹误差评估"],
        ["球最终位置", "39条有最后观测的像素/归一化坐标", "部分；未证实是球门线处或事件结束的最终位置"],
        ["Goal / no goal", "外部赛果CSV按clip ID并入39条", "仅标签整合；尚无已验证的视频自动判定"],
        ["罚球者背后机位", "标准片段中20个候选段，涉及14条视频", "部分；这是分类结果数量，不是真实出现次数"],
        ["罚球者右侧45度", "13个候选段，涉及12条视频", "部分；线条角度不直接证明相机位于右侧"],
        ["罚球者慢动作特写", "52个候选段，涉及29条视频", "部分；低绿地比例不能单独证明慢动作"],
        ["守门员背后慢动作", "49个候选段，涉及35条视频", "部分；其余画面归入该类，缺少明确拒识及慢放验证"],
        ["基础EDA", "有模块计数、质量摘要和旧绘图脚本", "部分；缺少基于最新真实联合表的可交付EDA"],
    ], [106, 188, 213])
    add("机位原表为49个文件的164段；按39个标准ID只读筛选后为134段。不是每条视频都必须有全部四类，但应检验实际存在的角度是否漏检或错分。守门员18条为整行联合规则接受数，不能把它当成三项特征各自的准确率。[E3、E5、E8]", "small")

    page("二 罚球者与新版步数的真实进展")
    add("已经完成的方法链", "sub")
    add("新版先通过球与人体的时序关联提出触球候选，再定位对应罚球者和助跑范围；结合姿态关键点、背景运动补偿和脚踝相对信号提出步态事件；最后将路线描述与临近踢球的节奏变化分开。与旧版直接依赖选中人物的最大脚踝运动相比，来源和失败状态更清楚。[E4]")
    table(["最新输出", "数量", "可作出的解释"], [
        ["连续助跑候选", "14/39", "观察到相对规律节奏；不等于完全无调整"],
        ["末段调整候选", "2/39", "Rashford与Aspas；只是本次罚球的候选类型"],
        ["类型未定但有计数", "15/39", "保留总数，不强行分段"],
        ["类型未定且无可靠计数", "8/39", "保留空值，不填0步"],
    ], [151, 67, 289])
    add("仍需解决的计步问题", "sub")
    add("候选事件源于脚踝运动峰谷，还未逐一对应“摆动—落脚—支撑”。主计数仍包含另列的触球保护区候选，其中可能既有最后支撑脚落地，也有踢球摆腿产生的峰谷。当前方式分类保留原总数，不会因为某人被判作碎步型就自动删减步数。")
    add("实际增强姿态对照只有30条，不是39条：19/30的名义计数改变，8/30相差至少2个候选。更大模型尚未被证明更准确。Cuadrado仍以5个候选报告；用户“约5步、4–6步可以接受”是单条粗看反馈，不是39条逐帧真值。[E4]")
    add("8条无可靠计数的具体原因", "sub")
    table(["片段", "当前拒绝输出的原因"], [
        ["Bacca；Brozovic；Fernandes", "存在竞争的人物—球接触身份，且独立触球证据不足"],
        ["Dier；西班牙对俄罗斯的Smolov", "独立触球证据不足，同时镜头补偿不完整"],
        ["克罗地亚对丹麦的Modric；Cheryshev", "没有通过当前规则的独立触球候选"],
        ["Kramaric", "触球候选暂可用，但关联姿态仅4帧，关键点覆盖不足，触球落在姿态缺口"],
    ], [200, 307])
    add("这些是算法证据链未通过，不表示原视频肉眼看不清。另有23条“类型未定”，也不能说成23条计步失败。COCO17只有踝点，不能从本模型输出宣称测到了锁踝能力、踝关节刚度或发力质量。", "small")

    page("三 守门员 足球 机位与几何")
    add("守门员  三项功能都有代码，但验证与输入仍需收尾", "sub")
    add("当前流程以球门区域高分辨率姿态、跨ID拼接、背景光流稳定及人体尺度归一化计算移动与扑救。39行中33行有数值，6行没有合格轨迹；18行规则接受、21行待查。触球帧39条全部标为未验证AI估计，身份种子36条亦非真人标注。质量门槛通过并没有消除这些输入来源限制。[E5]")
    add("足球  跟踪与最终位置仍是二维观测", "sub")
    add("代码结合YOLO sports-ball候选、白色/草地外观、脚部接触锚点及朝球门连续运动约束。全部39条保留轨迹，26条规则接受。搜索默认延伸到旧触球帧后32帧，约1.07秒；final_position取所选轨迹的最后一点。它可能因遮挡、出框或跟丢而提前结束，不能直接称为入网点或整段比赛结束位置。球速单位是px/s，不是m/s或km/h。[E6]")
    add("Outcome  赛果标签已经整合，自动识别未完成", "sub")
    add("球模块显式把 penalty_outcomes.csv 的 observed_outcome 传给结果函数，因此goal/no_goal是读取标签，而不是本轮追踪推断。表中记录来源为FIFA技术报告，但本次没有重新逐条核验外部报告。旧几何回退仅看默认矩形和接触标志，不能视为有效进球判定。若提交范围要求视频自动提取，该项必须补上或向老师明确说明范围。[E6]")
    add("机位  有四类启发式候选，尚无类别与时序验收", "sub")
    add("现有模块用HSV直方图差找切镜，再用绿地比例和Canny/Hough线条角度分四类。代码没有直接验证慢动作速度或明确的相机方位；普通画面也可能被归入慢放类。已有独立提取函数，但缺少面向标准39片段、带拒识和验证证据的完整独立使用入口。[E8]")
    add("俯视轨迹  工具准备充分，真实重建尚未落地", "sub")
    add("单应性模块已增加退化点检查、RANSAC误差与单位保护；39张线条候选图并不是39次完成球场标定。尚无真实地面标志点的语义匹配、镜头变化后的有效性检查和独立重投影验证。默认没有H时返回原像素坐标，不能称为俯视轨迹。旧EDA仅画起点到点球点的一条箭头，也不是随时间的实际曲线。[E2、E9]")
    add("Mask与Edge的贡献", "sub")
    add("已有HSV掩膜与边缘/直线候选实现。保留的mask消融记录显示35条非空计数没有增加、人物选择改变0条；它证明方法已尝试，不证明准确率提升。本次没有重新运行该消融。", "small")

    page("四 数据完整性与可复现性审阅")
    table(["发现", "本次核查证据", "影响"], [
        ["旧总表不是真实联合表", "49行；步数全1、结果全goal；旧pipeline手写人物、门将、球轨迹", "旧米制运动量与两幅EDA图不得进入正式结论"],
        ["标准ID与文件枚举混用", "原视频49个；标准39个均在；机位表把额外10个也算入", "全局目录扫描会重复计入点球"],
        ["触球时间跨模块未统一", "32个可比新版触球锚点中29个不同于旧标注；15个相差超过3帧", "步数、门将、球结果不能直接视为同一版本事件窗口"],
        ["主notebook与说明冲突", "单元166/172仍称人工确认；169/175未加探索运行标志", "来源误述；默认来源校验会阻止旧输入重跑"],
        ["新版分支尚未整合", "新步数只存独立JSON与新查看notebook", "旧主入口不能复现一张最新完整特征表"],
        ["环境声明不一致", "总requirements与阶段实验记录的NumPy/SciPy/OpenCV版本不同", "尚未证明干净环境安装并端到端运行成功"],
    ], [111, 247, 149])
    add("本次实际执行的检查", "sub")
    add("重新运行146项选定测试并通过；重新检查78份基础/增强结果的输入保留、计数分区、数值重放和代码哈希；39份视频源文件/播放副本哈希一致。验证输出单独保存在本次审阅目录，没有覆盖旧实验验证文件。[E10]")
    add("先尝试标准unittest全目录发现时，131个可运行测试通过，另有1个测试模块因缺少pytest而导入失败；该命令还不能自动收集全部函数式测试。随后执行项目的选定测试入口得到146通过。完整pytest测试集未能复跑，因此本报告不声称“全仓库测试全通过”。")
    add("这些检查证明程序逻辑和保存结果的一致性，不证明真实落脚、人物身份、轨迹终点或四类机位识别正确。没有从空白环境重跑所有原始视频，也没有在本次审阅中重新逐帧观看39段视频。")
    add("标注来源", "sub")
    add("confirmed_step_count和confirmed_landing_frames的39行均为空；ground_truth_eligible全为False。旧文件名中的manual或confirmed不能作为人工复核证明。本次没有添加或补造复核人、时间、落脚帧或任何准确率。", "small")

    page("五 交付状态与建议收尾顺序")
    add("按提交物审阅", "sub")
    table(["提交物", "当前状态", "完成条件"], [
        ["代码与数据", "模块、权重、片段、README和依赖文件已存在；整合未完成", "统一入口与版本；干净环境复跑；移除正式入口对演示数据的依赖"],
        ["正式项目报告", "已有方法和来源说明；本地未见符合全部要求的最终英文论文PDF", "文献、数据、EDA、2–3个研究问题、新技术、代码结构及限制齐备"],
        ["10分钟演示视频", "当前项目目录未发现可确认的最终演示成片", "完成成片与播放检查；不把复核视频当作演示视频"],
        ["课程平台上传", "本次未检查，不能确认", "由实际上传记录确认，不凭本地文件推断"],
    ], [98, 209, 200])
    add("原说明列明报告正文最多25页，字号不小于10号Times New Roman，要求英文连续论述；视频10分钟；截止日期在原文件写为10月30日23:59。本次没有核验老师是否后来更新通知，亦没有代为上传。[E1]", "small")
    add("优先级1  先让结果可信且不混版", "sub")
    add("将旧演示总表和图明确隔离出正式分析入口；统一39个标准ID、视频绝对帧号、选人版本及触球锚点；修正主notebook残留来源描述和运行命令，再生成新的联合表。此次仅报告问题，没有修改这些分析文件。")
    add("优先级2  补齐老师明确要求但尚未落地的功能", "sub")
    add("优先完成真实俯视轨迹、四类机位的独立输出与验证，以及视频自动outcome的可解释证据链。外部赛果可以作为核对参考，不应再称作自动提取结果；球终点必须区分“最后看见”“球门线附近”和“事件结束”。")
    add("优先级3  完成独立验证并保留失败", "sub")
    add("先确定步数边界与落脚定义，覆盖四场比赛和主要错误类型做独立复核。人工参考用于评估，不能替换生产提取过程。已有全部片段都参与开发时，不应宣称某场此前未见；后续冻结规则后的留出评估要披露开发暴露情况。按任务报告计步误差、方向一致性、球定位误差及机位边界误差；不把无输出样本删掉后计算好看的整体率。")
    add("优先级4  基于真实联合表完成研究与交付", "sub")
    add("EDA至少报告各特征的可用性与失败分布、罚球动作的描述性差异，以及特征与赛果的探索性关系。比较时逐图注明实际样本数、单位、筛选规则和版本，不把二维代理量解释为真实生物力学或因果关系。之后冻结环境，完成英文论文与视频。数据量少不作为本次主要薄弱点。")

    page("六 证据索引与验收口径")
    add("项目根目录：C:/Users/30605/PycharmProjects/JupyterProject。以下路径相对此根目录；单元号为notebook从0开始的JSON索引。报告数值是2026年10月9日快照。", "small")
    table(["ID", "本地证据", "支撑的结论"], [
        ["E1", "docs/ind5003_vision_ii_project_requirements.pdf，第4–7页", "原任务与交付要求；SHA256与Downloads原件相同"],
        ["E2", "src/pipeline.py；src/eda_analysis.py；data/processed/penalties_features.csv；outputs/plots/", "49行示例结果与不适合作为实证的旧EDA"],
        ["E3", "configs/penalty_slicing_manifest.yaml；data/raw_clips/；data/processed/camera_timings.csv", "39标准ID、49文件、164原始及134标准机位段"],
        ["E4", "outputs/style_aware_steps_2026-10-09/下的classified/summary.json与model_comparison.json；src/style_aware_steps.py", "31条计数、14/2/23类型、增强姿态分歧"],
        ["E5", "data/processed/penalty_goalkeeper_features.csv；src/goalkeeper_pipeline.py；data/annotations/penalty_goalkeeper_track_reviews.csv", "33条计算值、18规则接受、来源限制"],
        ["E6", "data/processed/penalty_ball_outcome_features.csv；src/ball_pipeline.py；src/ball_outcome.py；data/annotations/penalty_outcomes.csv", "轨迹26规则接受、最后观测点、outcome标签合并"],
        ["E7", "notebooks/vision2_penalty_analysis.ipynb，单元153、166–175；requirements.txt；requirements-phase-experiment.txt", "主入口陈旧描述、重跑缺口、环境差异"],
        ["E8", "src/camera_angles.py", "HSV切镜、颜色与线条启发式，缺少慢动作/方位验证"],
        ["E9", "src/homography.py；outputs/mask_optimization_2026-10-08/下的pitch_geometry/index.json与summary.json", "几何封装与39项未标定记录、mask消融范围"],
        ["E10", "outputs/completion_audit_2026-10-09/下的evidence.json与current_validation.json", "本次读取快照与146测试、78重放、39哈希检查"],
    ], [33, 326, 148])
    add("建议的最终验收口径", "sub")
    add("“已实现”要求能从声明输入经脚本产生对应特征；“可用结果”要求保留覆盖、来源与不确定性；“已验证”要求有独立参考及误差统计；“可提交”还要求统一版本、可复现环境、有效EDA和规定交付物。当前项目达到实验原型与部分候选结果阶段，尚未达到整体可提交状态。")
    add("不提供单一总完成百分比：九项特征、四类机位、验证、EDA与提交材料并非等量工作。用任意权重拼成80%或90%会掩盖尚未实现的必需项；本报告改为逐项状态及明确分母的覆盖率。", "small")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=44, leftMargin=44,
                            topMargin=43, bottomMargin=47, title="Vision II 项目完成度审阅报告",
                            author="项目审阅", subject="要求覆盖、结果质量与可复现性审阅")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
