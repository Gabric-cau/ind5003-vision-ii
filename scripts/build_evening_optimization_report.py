"""Generate the Chinese experimental handoff, not the final course submission."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Image,PageBreak,Spacer,Table,TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from PIL import Image as PILImage
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_vision2_completion_report import p,FONT,BLUE,MUTED
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
OUT=ROOT/"output/pdf/vision2_optimization_experiments_2026-10-09.pdf"
story=[]


def add(t,style="body"):story.append(p(t,style))
def page(t):
    if story:story.append(PageBreak())
    add(t,"heading")
def table(headers,rows,widths):
    data=[[p(t,"headcell") for t in headers]]+[[p(str(t),"cell") for t in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),
      ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F1F4F7")]),
      ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),
      ("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),
      ("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,9)])
def pic(path,width=507,maxheight=400):
    with PILImage.open(path) as im:w,h=im.size
    scale=min(width/w,maxheight/h)
    story.extend([Image(str(path),width=w*scale,height=h*scale),Spacer(1,8)])
def footer(c,doc):
    c.setFont(FONT,8.5);c.setFillColor(MUTED)
    c.drawString(44,29,"Vision II 优化实验验收   |   2026年10月9日")
    c.drawRightString(A4[0]-44,29,str(doc.page))


def build():
    m=json.loads((RUN/"integration/metrics.json").read_text(encoding="utf-8"))
    assert m["tests"]["passed"] and m["tests"]["total_tests"]==168
    assert (m["baseline_count_available"],m["v4_count_available"],m["foot_parameter_stable"])==(31,39,1)
    assert m["gk_guard"]["guarded_qa"]==15
    add("Vision II 优化实验报告","title")
    add("面向19:00验收  |  39条标准点球片段  |  基线保留、独立实验版","small")
    add("本轮结论","sub")
    add("有优化空间，而且已做真实视频实验。最直接的进展是补齐原8条缺失的步数候选、用球门区域自动定位守门员、修复轨迹中断与可视化误导，并将真实39条结果汇总为可追溯数据。没有证据支持宣称项目已100%正确或全部完成。")
    table(["模块","本轮结果","采用边界"],[
      ["助跑步数","31/39 → 39/39有候选总数","输出覆盖100%，不是准确率；1条明确仅覆盖助跑后段"],
      ["左右脚","39条名义标签；仅1条通过严格参数稳定性检查","不采用为已验证提升；原35条候选亦非人工真值"],
      ["守门员","球门ROI+外观：29条原QA接受；完整窗口检查后15条","不同QA口径，不能把15/39或29/39称为准确率"],
      ["球轨迹","13条困难样本：4组连续性QA为7/6/7/5","短窗口优于盲目延长；未重新推理其余26条"],
      ["机位分段","39条完整覆盖，187个候选区间","切镜及四类机位仍未验证；慢动作不强行标注"],
      ["地面俯视/落点/outcome","保留未知；完成几何与记分牌可行性检查","尚未解决，未用外部标签冒充自动识别"],
      ["真实联合结果与EDA","已生成39条联合JSON及诊断图","不再使用49行示例总表；尚非最终研究因果结论"],
    ],[85,215,207])
    add("如何读本报告","sub")
    add("候选覆盖率＝有输出的比例；QA＝规则筛选；参数稳定性＝换固定参数后是否一致；准确率＝与独立真实标签比较。前三者均不等于第四者。用户没有逐片人工复核，助手看过诊断图也不构成人工真值。当前不提供虚构的准确率、步数MAE或总完成百分比。")

    page("1. 触球定位与步数：从失败原因入手")
    add("为什么人看得清，程序却失败","sub")
    add("问题不只是球小或画面不清。流程必须先确定正片镜头、踢球者身份、静止球与触球时刻，再在正确时间段内计步。把离场、回脚或重复轨迹当成踢球过程，后面的计步会在错误窗口里稳定地产生错误。")
    table(["实验","改动","发现"],[
      ["提高检测密度","困难8条每3帧检测球","单独加密不足；Smolov帧129候选出现在击球后，未采用"],
      ["静止球离开窗口","39条同密度对照；限制最近脚踝搜索范围","减少静止球已离开后回脚被选中的机会；并不保证每个窗口正确"],
      ["自动区域放大","由人物接近证据与球门几何构建ROI","8条中7条生成接触候选；无需按姓名硬写触球帧"],
      ["中型姿态+首次离球","Brozovic分块检测；同一动作优先早期因果候选","全图仍失败；高分帧89为离场；改为暂定帧34，仍需复核"],
    ],[96,187,224])
    add("补齐的8条候选（不是人工确认步数）","sub")
    names={"col_eng_p09_bacca_col7":"Bacca / COL–ENG","col_eng_p10_dier_eng4":"Dier / COL–ENG",
      "cro_den_p04_kramaric_cro9":"Kramaric / CRO–DEN","cro_den_p06_modric_cro10":"Modric / CRO–DEN",
      "esp_rus_p02_smolov_rus10":"Smolov / ESP–RUS","esp_rus_p08_cheryshev_rus6":"Cheryshev / ESP–RUS",
      "rus_cro_p02_brozovic_cro11":"Brozovic / RUS–CRO","rus_cro_p05_fernandes_rus2":"Fernandes / RUS–CRO"}
    table(["片段","候选总数","参数变动范围","触球规则状态"],
      [[names[r["clip_id"]],r["candidate_count"],"–".join(map(str,r["parameter_range"])),r["contact_status"]] for r in m["new_count_rows"]],
      [215,62,104,126])
    add("范围来自固定峰谷阈值×触球锚点的敏感性分析，不是统计置信区间，也不是“真实步数应该在此范围”。Brozovic姿态从帧18开始，2步只能解释为被跟到的后段；不能宣称完整助跑只有2步。", "small")

    page("2. 步数分类与踢球脚：哪些结论应保留")
    add("整体计步与末段调整分开","sub")
    add("本轮保留原来的方式感知计数：连续助跑按整体读；末段调整另列分区；类型不确定但总数可算时保留总数。前段弧线/横移是路线背景，不自动等于末段脚步变化；不因为觉得球员应该是5步，就把候选事件裁成5步。")
    table(["指标","实验结果","含义"],[
      ["原有31条计数","28条不变；2条减少1；1条增加1","不是与真值的误差；只说明换触球证据后的变化"],
      ["39条方式候选","13连续、4末段调整、22未定","每次罚球的描述，不是球员永久风格"],
      ["触球保护区","仍计入主候选总数，并单列","支撑脚落地与摆腿峰谷尚未逐一分清"],
      ["脚踝控制","没有可信的生物力学控制指标","2D关键点不能直接测量脚踝力矩、击球接触质量或关节控制能力"],
    ],[106,178,223])
    add("左右脚压力测试：发现旧方法的薄弱点","sub")
    add("原方法比较触球附近左右脚踝相对髋中心的最大帧间位移，以较大者作为踢球脚。问题是支撑脚、摆腿回收、左右关键点错配、遮挡及触球锚点偏差均可能改变最大值。这个方法会输出标签，但标签未必稳定。")
    add("本轮在接触区间的早/名义/晚3个锚点，分别使用3帧和5帧窗口。严格保留条件为所有变体标签一致、无unknown，且每次左右运动优势比均≥1.25。39条名义输出中仅1条满足全部条件。这个1/39是稳定性筛选通过数，不是只有1条踢球脚正确。")
    add("本轮决定","sub")
    add("不把39条名义左右脚标签作为成功率提升发布；与基线标签及稳定性标志分列保存。下一步更合理的研究是使用球—脚空间接触、支撑脚持续落地与摆腿方向的联合证据，并用少量独立复核先测错因。当前没有完成这一替代模型。")
    add("变化的3条","sub")
    add("Schone（CRO–DEN）7→6；Golovin（ESP–RUS）4→5；Modric（RUS–CRO）9→8。其余原有候选计数不变。以上都是模型版本对照，不是纠正后的真实步数。", "small")

    page("3. 守门员：消除依赖，也避免虚高通过数")
    add("新流程","sub")
    add("白色HSV掩膜→形态学/边缘/Hough线段→球门框候选→球门内人物几何约束→ROI放大姿态→躯干HSV外观相似度→断轨限制→运动特征→观察完整性检查。最终ROI路线不读旧AI身份种子或旧AI触球CSV，触球来自本轮独立候选。")
    table(["阶段（均39条）","规则接受","为什么不能直接宣布改善"],[
      ["旧缓存移除身份种子","14→18（限制断轨后）","可能选中裁判，否决为最终方案"],
      ["球门ROI新姿态","36；38条有身份候选","扑救后会跳到裁判/看台人；原QA放过错误"],
      ["躯干外观+区域/断轨限制","29","减少部分跳转，但同时截断一些真实扑救"],
      ["完整1秒窗口限制","15；16条窗口完整","23条扑救类别unknown；还有1条其他QA不通过"],
    ],[186,109,212])
    add("为何更严格后通过数反而少了","sub")
    add("这是修改“什么证据足够支持整段判断”，不是模型准确率从29降到15。固定触球后1秒要求至少80%帧有观测、首尾距窗口边界不超过0.1秒、内部缺口不超过0.2秒。未观察到后段就不能断言整次扑救居中、低扑或高扑。这些阈值仍是工程选择，未用真实标签标定。")
    pic(RUN/"goal_appearance_v2/qa_gap_safe/col_eng_p08_trippier_eng12.jpg",maxheight=100)
    add("Trippier片段的历史外观版诊断图：图头仍显示旧候选center/review=False。实际只跟到帧78–92，完整窗口要求到108；新结果保留旧候选，但最终类别改为unknown。末帧无标记表示没有该时刻观测，不再拿过去坐标涂到未来帧上。", "small")
    add("仍有风险","sub")
    add("球衣颜色相近、人物交叠、躯干姿态剧变、镜头运动均可能造成身份混淆或提前断轨。RUS–CRO Ignashevich的诊断图仍提示晚段裁判混淆风险；因此即使通过新QA也不是确认正确。左右方向是画面坐标，不是守门员解剖学左右；高度是2D身体位移代理，不是实际扑救高度。")

    page("4. 足球：更长窗口不是自动更好的轨迹")
    add("实验设计","sub")
    add("只选择基线13条review_required视频，比较全图/自动走廊ROI与触球后32/60帧的四种组合。保持同一本地YOLO11n检测器；用独立触球候选与脚踝位置约束起点，限制在同镜头内。赛果字段不进入预测；整合阶段才读取外部标签。")
    table(["方案","原困难组内连续性QA通过","解释"],[
      ["全图，32帧","7/13","作为固定短窗实验参考，非最终精度结论"],
      ["全图，60帧","6/13","延长后更容易接上远处白色杂点"],
      ["自动ROI，32帧","7/13","未在此QA指标上优于全图短窗"],
      ["自动ROI，60帧","5/13","ROI也不能阻止所有后续误关联"],
    ],[138,166,203])
    add("新增的工程保护","sub")
    add("所有运动假设过期时，旧搜索会对空集合求max而崩溃，现已修复。实验选项保留已观测的短轨迹，避免因后续无检测而把真实前缀丢掉；同时检查最大观测间隙，超过0.25秒的轨迹不通过连续性QA。不插值伪造消失后的球。")
    add("失败实验也有价值","sub")
    add("Cuadrado长窗口候选曾跨越48帧缺口连接到较远点，旧QA可能通过。这个延长不能当成发现了更完整飞行轨迹；新连续性条件会拒绝。提高检测密度、拉长时间或做mask必须结合关联逻辑，而不是只看点数变多。")
    add("范围限制","sub")
    add("13条中Brozovic在该实验运行时没有可用触球锚点，因此弃权；没有把后来帧34候选重新用于球推理。其他26条基线已接受视频本轮未重新推理。因此不能把本轮7条加到原26条后就宣称33/39成功，更不能称为准确率。")
    add("尚未解决","sub")
    add("最后观测到的像素位置不是射门最终落点；2D像素速度不是m/s；球的腾空高度不能用球场地面单应性直接还原。遮挡后是否入网仍缺独立证据。所有自动outcome保留unknown。")

    page("5. 俯视、机位和outcome：可行性与边界")
    add("Mask与edge用在了哪里","sub")
    add("本轮在球门定位中实际使用了白色HSV阈值、inRange二值掩膜、形态学、Canny边缘与Hough线段。这些课堂方法帮助限制目标搜索空间，减少草地或背景干扰。初始球门框候选从5条、9条发展到36/39；该数字来自既定帧的候选检测，不是36条标定成功。")
    add("为什么还不能生成可信的俯视跑动","sub")
    add("球门框在竖直平面，球员脚底跑动在草坪平面，两者不能共用一个平面变换。真正地面单应性需要匹配到已知球场坐标的充分对应点、明确坐标约定及重投影检查。当前没有39条逐片验证的地面标定；因此不生成看似漂亮却没有几何依据的俯视轨迹或米制跑速。")
    table(["项目要求","本轮做了什么","尚缺什么"],[
      ["四类机位的时间段","只对标准39条重新切镜；阈值30/45对照；187区间连续覆盖所有帧","切镜真值验证；背后/45度右侧等方位证据；慢动作证据"],
      ["慢动作分类","旧草地/线条启发式标签保留为候选，verified与slow_motion标为unknown","不能因近景或画面稳定就断言慢放；需时间运动证据"],
      ["自动outcome","分层抽取8条片段，各看开头/75%/末尾，检查记分牌变化可见性","24个样本帧未见可读数字比分变化对；没有完成OCR进球判定"],
      ["真实球终点","保留最后观测点及非事件终点标记","球是否越过门线、被扑/出界的持续时序证据"],
    ],[105,218,184])
    add("不要过度推断","sub")
    add("分段帧覆盖100%只意味着没有丢掉时间区间，不等于100%切镜正确。8条抽样未见比分变化，也不证明全部视频的每一帧都没有可读记分牌。现有26进、13未进是来源字段注明的外部赛果标签，本轮没有重新核验外部来源，更没有把它们当视频模型输出。")

    page("6. 真实39条联合数据与描述性EDA")
    add("已经移除研究入口中的示例数据误用","sub")
    add("新联合结果逐条按标准clip_id连接步数、触球、左右脚、守门员、球实验、机位与赛果来源；每条都保留human_verified=false与accuracy_validated=false。旧49行示例总表未删除，但本轮汇总脚本完全不读取它。")
    pic(RUN/"integration/actual39_diagnostics.png",maxheight=345)
    add("可支持的描述","sub")
    add("39条候选计数分布不同，6步候选最多（12条）；方式未定仍有22条；原31个可配对计数中28条不变。右下图仅比较13条困难视频的规则通过数，不能直接推广为全数据精度。高计数14/15可能包含密集候选或误检测，未按主观期望强行削减。")
    add("暂不支持的研究结论","sub")
    add("没有依据说“末段调整让进球率提高”“弧线助跑没用”或“脚踝控制更强”。我们没有独立准确的动作标签，也没有控制球员、守门员、角度等混杂因素。外部26/39进球只能作为样本标签描述。该图是有效的实际数据诊断EDA，不是完整的战术因果研究。")
    add("可追溯输出","sub")
    add("integration/canonical_results_39.json 为联合明细；metrics.json 为报告数字来源；actual39_diagnostics.png 为可复用图。步数参数范围、守门员前后QA、未重新推理的球样本均明确分列，不拼成一个虚假的“总成功率”。", "small")

    page("7. 验证、代码来源与复现交付")
    add("已运行的检查","sub")
    table(["检查","结果","不能证明什么"],[
      ["选定合成/回归测试","168项通过（98历史+70新增/相关）","不是整个仓库pytest全量运行；不是视频精度"],
      ["步数确定性重放","39份通过分区守恒、来源与重放检查","不能证明候选等于真实落脚"],
      ["原始视频SHA256","39份与此前记录一致","证明视频未被改写，不是识别正确"],
      ["联合结果一致性","39个唯一ID，来源/未知/标签分列","没有独立真值；未做锁定测试集评估"],
      ["可视检查","助手查看接触、守门员、球及比分诊断图","不是用户复核、不是逐帧全覆盖标注"],
    ],[114,182,211])
    add("方法和代码从哪里来","sub")
    table(["层次","本轮来源/用途"],[
      ["预训练模型","现有Ultralytics YOLO11n检测/姿态与YOLO11m姿态权重；没有重新训练或声称在SoccerNet上训练"],
      ["通用算子","OpenCV的HSV/inRange、形态学、Canny、Hough、视频解码；NumPy/SciPy数值与信号处理"],
      ["本项目组合逻辑","触球窗口与首次离球规则、人物关联、方式感知计步、球衣外观约束、轨迹完整性与来源审计"],
      ["非本轮实现","没有新的端到端射门网络、真实3D球轨迹、脚踝生物力学模型或可靠自动进球判断"],
    ],[112,395])
    add("复现入口与环境","sub")
    add("在项目根目录，使用 C:/Users/30605/Virtual_Environment/myenv/Scripts/python.exe 运行 scripts/vision2_evening.py inspect、validate 或 bundle。inspect只读；后两者重建本实验验证与汇总。详细命令及推理脚本见实验README。")
    add("requirements-evening-experiment.txt记录真实运行版本；environment.json记录具体导入位置。使用的是myenv与项目.venv补充路径的混合环境，启动器有迁移警告；没有完成全新环境安装测试。重推理脚本仍有固定来源目录，未完成从原视频一键重放所有消融版本。", "small")
    add("保留与限制","sub")
    add("原视频、旧结果与原大Notebook保留；最终源码/测试快照单独保存。部分旧实验只有输出、参数与哈希，没有完整历史源码快照，因此不声称每个中间版本都能原样重现。本轮是在同一批样本上探索方法，有过拟合当前案例风险。最终英文课程报告、演示视频及干净的一键提交包仍未在本轮完成。", "small")

    page("8. 验收清单与下一步优先级")
    table(["优先级","建议","验收条件"],[
      ["P0","先复核8条新增步数与左右脚争议；标注触球邻域、有效助跑起点和计步定义","记录真实复核人/时间；不把助手输出复制为ground truth"],
      ["P0","修正守门员身份连续性，先定位仍选中裁判的轨迹","独立身份复核+断轨率/切换次数；不只看QA接受数"],
      ["P1","基于球—脚接触与支撑脚约束替代单纯最大脚踝位移","先做锁定样本对照；允许unknown；报告错误类型"],
      ["P1","完成球场地面标定或明确不可标定片段","对应点来源清晰、重投影误差可查、地面与球门平面区分"],
      ["P1","自动outcome与球终点单独建事件证据链","不读赛果标签决定预测；遮挡/镜头切换保留未知"],
      ["P2","统一Notebook、所有阶段路径与环境；再做研究EDA与最终材料","从原视频到最终表可复现，英文报告与视频满足课程要求"],
    ],[48,229,230])
    add("本轮建议采用与暂不采用","sub")
    add("建议采用：独立版本目录、真实39条联合表、显式unknown、源视频哈希、球搜索空集合修复、观察缺失不外推，以及新的诊断图显示规则。可以继续使用但要标实验候选：新触球/步数、球门ROI/外观跟踪与短窗球轨迹。暂不采用为正式结论：39条左右脚名义标签、36条守门员旧QA“成功”、延长后球终点、未标定的俯视图和外部赛果冒充自动识别。")
    add("核心证据位置（相对项目根目录）","sub")
    add("outputs/optimization_evening_2026-10-09/ 为本轮实验根目录。优先查看 README.md、integration/metrics.json、integration/canonical_results_39.json、validation.json、visual_observations.json；之后按逐片source路径打开对应JSON及诊断图。", "small")
    add("关键代码：src/kick_contact.py、src/ball_roi_probe.py、src/goal_geometry.py、src/person_appearance.py、src/goalkeeper_pipeline.py、src/goalkeeper_observation_guard.py、src/ball_pipeline.py；汇总入口为scripts/vision2_evening.py。各阶段脚本、参数与失败记录见实验README。", "small")
    add("验收总结","sub")
    add("这轮交付的是“做过实验、有真实改动、保留负结果且能追溯的优化版本”。它补齐了候选覆盖与数据整合，也暴露出左右脚稳定性、守门员身份、终点/outcome和标定方面的关键缺口。下一阶段应以独立验证和最主要错因修复为目标，不以让每个字段都非空为目标。")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=47,
        title="Vision II 优化实验报告",author="Vision II 项目",subject="不完整模块优化、对照实验与证据边界")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT)


if __name__=="__main__":build()
