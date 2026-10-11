"""Freeze candidate-only camera shortcut, then evaluate against explicit user review.

No human label enters fresh pose, approach extraction or candidate prediction.
This is a development-cohort comparison, not a held-out accuracy estimate.
"""
from pathlib import Path
from collections import Counter
import json,sys,html,os,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kicking_foot_redetection import camera_scoped_approach
RUN=ROOT/'outputs/goal_right_foot_redetection_2026-10-11/run_v2'
GALLERY=ROOT/'outputs/goal_right_camera_review_2026-10-11'
ANNOTATION=ROOT/'data/annotations/kicking_foot_user_review_goal_right_2026-10-11.json'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows=load(RUN/'summary.json')['rows'];follow={r['clip_id']:r for r in load(RUN/'followthrough_analysis.json')['rows']}
    baseline={r['clip_id']:r for r in load(GALLERY/'selection.json')['selected']}
    predictions=[]
    for r in rows:
        cid=r['clip_id'];prior=camera_scoped_approach(r['approach_heuristics'],'goal_right')
        pose=follow[cid]['followthrough_consensus_candidate']
        predictions.append(dict(clip_id=cid,experimental_kicking_foot_candidate=prior['candidate'],method=prior['method'],
            requires_review=True,anatomical_pose_crosscheck_candidate=pose,
            pose_vs_approach_conflict=pose!='unknown' and pose!=prior['candidate'],
            shoe_ball_contact_verified=False,reviewer_labels_used_for_prediction=False))
    save(RUN/'camera_scoped_predictions.json',dict(cohort_gate=list(baseline),camera_gate='AI-reviewed goal-right primary shot layout only',
        policy='Candidate prior, not automatic globally accepted anatomical label',rows=predictions,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/kicking_foot_redetection.py']}))
    # Only after predictions are produced do we read the user's reference labels.
    annotation=load(ANNOTATION);refs={r['clip_id']:r['kicking_foot'] for r in annotation['reviews']}
    assert set(refs)==set(baseline)=={r['clip_id'] for r in rows}
    records=[]
    methods=['old','previous_experiment','fresh_normal_motion','fresh_mirrored_motion','strict_contact_fusion',
             'literal_screen_side','projected_goal_axis','followthrough_consensus','pose_axis_crosschecked','camera_scoped_shortcut']
    for r,p in zip(rows,predictions):
        cid=r['clip_id'];f=follow[cid];b=baseline[cid]
        values=dict(old=b['old_foot'],previous_experiment=b['new_foot'],fresh_normal_motion=r['normal_evidence']['motion_candidate'],
            fresh_mirrored_motion=r['mirrored_evidence']['motion_candidate'],strict_contact_fusion=r['automatic_candidate']['candidate'],
            literal_screen_side=r['approach_heuristics']['screen_x_candidate'],projected_goal_axis=r['approach_heuristics']['goal_axis_candidate'],
            followthrough_consensus=f['followthrough_consensus_candidate'],pose_axis_crosschecked=f['approach_crosschecked_candidate'],
            camera_scoped_shortcut=p['experimental_kicking_foot_candidate'])
        records.append(dict(clip_id=cid,user_reviewed_kicking_foot=refs[cid],methods=values,
                            shortcut=p,contact_bracket=r['contact_bracket_frames']))
    comparison={}
    for method in methods:
        comparison[method]=dict(correct=sum(r['methods'][method]==r['user_reviewed_kicking_foot'] for r in records),
            incorrect=sum(r['methods'][method] not in ('unknown',r['user_reviewed_kicking_foot']) for r in records),
            unknown=sum(r['methods'][method]=='unknown' for r in records),total=len(records))
    evaluation=dict(cohort=len(records),reference_source=str(ANNOTATION),reference_sha256=sha(ANNOTATION),
        reference_distribution=dict(Counter(refs.values())),comparisons=comparison,records=records,
        held_out_test=False,development_cohort=True,global_accuracy_estimate=None,
        warning='Methods were explored on this camera cohort after user feedback. Matches on these 10 clips do not establish generalization or all-39 accuracy.',
        official_predictions_overwritten=False,steps_goalkeeper_ball_outcome_unchanged=True)
    save(RUN/'evaluation.json',evaluation)
    shutil.copy2(ANNOTATION,RUN/'user_review_reference.json')
    for p in [Path(__file__),ROOT/'scripts/analyse_goal_right_followthrough.py',ROOT/'tests/test_kicking_foot_redetection.py']:
        shutil.copy2(p,RUN/'source_snapshot'/p.name)
    # Preserve the exact inference module, including before later functions were added.
    expected=next(v for k,v in load(RUN/'protocol.json')['source_sha256'].items() if k.endswith('kicking_foot_redetection.py'))
    inference_text=(ROOT/'src/kicking_foot_redetection.py').read_text(encoding='utf-8').split('def followthrough_evidence(')[0].rstrip()+'\n'
    assert hashlib.sha256(inference_text.encode()).hexdigest()==expected
    (RUN/'source_snapshot/kicking_foot_redetection.py').write_bytes(inference_text.encode())
    # This explicit file is the current reanalysis source; inference snapshots remain intact.
    shutil.copy2(ROOT/'src/kicking_foot_redetection.py',RUN/'source_snapshot/kicking_foot_redetection_current_reanalysis.py')
    def render(base):
        rel=lambda p:Path(os.path.relpath(p,base)).as_posix()
        text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>右侧球门机位：重新判脚实验</title><style>body{font:16px/1.65 system-ui;background:#15202b;color:#eee;max-width:1250px;margin:20px auto;padding:16px}article{background:#243443;padding:18px;margin:22px 0;border-radius:10px}video,img{max-width:100%;width:100%}a{color:#9cdbff}table{border-collapse:collapse;width:100%}th,td{border:1px solid #647485;padding:6px}.warn{color:#ffd0a8}h2{font-size:20px}</style><h1>球门在右侧的10条视频：判脚重新检测</h1><p>fresh YOLO11m pose、人物区域放大、镜像后同时恢复像素坐标与左右关节通道；重新比较踢后伸展与相对射门轴的接近侧。没有按球员姓名硬编码脚，也没有全组互换左右。</p><p class="warn">下列结果是开发样本对照，不是独立测试准确率。你的复核：Pivarić 左脚，其余9次右脚。算法预测与用户复核分别显示；新捷径依赖未标定二维几何，仍需复核，仅限这10条主镜头。</p><p>简单画面左／右规则把10条全判右脚，错过Pivarić。相对球—球门射门轴的接近侧给出9个与复核一致的候选、1个未知；Badelj 缺少自动球门几何，因此单列为较弱的画面方向后备候选。</p>'''
        text+='<table><tr><th>方法</th><th>与本次复核一致</th><th>不一致</th><th>未知</th></tr>'
        names=dict(old='旧版',previous_experiment='上一实验版',fresh_normal_motion='新模型原视图：最大脚运动',strict_contact_fusion='严格脚／球／镜像一致性',literal_screen_side='简单画面左右捷径',projected_goal_axis='相对射门轴的接近侧',pose_axis_crosschecked='伸脚方向与接近侧同时一致',camera_scoped_shortcut='限定机位捷径（含1条较弱后备）')
        for key,name in names.items():
            v=comparison[key];text+=f'<tr><td>{name}</td><td>{v["correct"]}</td><td>{v["incorrect"]}</td><td>{v["unknown"]}</td></tr>'
        text+='</table><p>严格脚／球判定仍有大量弃权，不能为了出结果把它们藏起来。接近侧捷径的10/10匹配不等于全项目准确率100%。</p>'
        for record in records:
            cid=record['clip_id'];p=record['shortcut'];m=record['methods']
            status='相对射门轴的接近侧候选' if p['method']=='projected_goal_axis_approach_prior' else '缺球门几何：较弱画面方向后备'
            conflict='；姿态伸脚证据与接近侧冲突，必须复核' if p['pose_vs_approach_conflict'] else ''
            speeds=''.join(f'<button onclick="this.closest(\'article\').querySelector(\'video\').playbackRate={v}">{v}×</button> ' for v in (.25,.5,1))
            text+=f'<article id="{cid}"><h2>{cid}</h2><p>用户复核：<b>{record["user_reviewed_kicking_foot"]}</b>；限定机位新候选：<b>{m["camera_scoped_shortcut"]}</b>（{status}{conflict}）。<br>旧版：{m["old"]}；上一实验版：{m["previous_experiment"]}；严格姿态／接触融合：{m["strict_contact_fusion"]}。<br>脚／球观察区间：源帧{record["contact_bracket"][0]}–{record["contact_bracket"][1]}，不是人工确定的精确触球帧。</p><video controls preload="none" poster="{rel(GALLERY/"frames"/(cid+".jpg"))}" src="{rel(GALLERY/"videos"/(cid+".mp4"))}"></video><p>{speeds}<a href="{rel(GALLERY/"videos"/(cid+".mp4"))}" download>下载视频</a></p><details><summary>新检测脚部关键点证据（L/R也是模型候选，黄圈为静止球位置参考）</summary><img loading="lazy" src="{rel(RUN/"evidence"/(cid+".jpg"))}"></details><a href="{rel(RUN/"results"/(cid+".json"))}">原始重新检测结果</a></article>'
        text+=f'<p><a href="{rel(RUN/"evaluation.json")}">逐条对照JSON</a> · <a href="{rel(RUN/"user_review_reference.json")}">用户复核来源</a> · <a href="{rel(RUN/"protocol.json")}">推理协议</a></p></html>'
        return text
    (RUN/'index.html').write_text(render(RUN),encoding='utf-8')
    backup=GALLERY/'index_before_foot_retest.html'
    if not backup.exists():shutil.copy2(GALLERY/'index.html',backup)
    # Put supporting artefacts inside the gallery server's narrow read-only root.
    local=GALLERY/'foot_retest';local.mkdir(exist_ok=True)
    for name in ['evaluation.json','user_review_reference.json','protocol.json']:
        shutil.copy2(RUN/name,local/name)
    for folder in ['results','evidence']:
        shutil.copytree(RUN/folder,local/folder,dirs_exist_ok=True)
    gallery_html=render(GALLERY).replace('../goal_right_foot_redetection_2026-10-11/run_v2/','foot_retest/')
    (GALLERY/'index.html').write_text(gallery_html,encoding='utf-8')
    save(RUN/'release_checks.json',dict(cohort_records=len(records),fresh_pose_records=sum(len(load(RUN/'poses'/f'{r["clip_id"]}.json')) for r in records),
        user_labels_separate_from_predictions=True,previous_gallery_preserved=str(backup),canonical_predictions_unchanged=True,
        source_snapshots_preserved=True,inference_module_snapshot_sha256=sha(RUN/'source_snapshot/kicking_foot_redetection.py'),global_accuracy_estimate=None))
    print(json.dumps(comparison,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
