"""Export a separate display layer; never replace automatic SAM2 evidence."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/goalkeeper_sam2_update_2026-10-11/results_v4_release'
REVIEWS = ROOT / 'data/annotations/goalkeeper_user_review_2026-10-11.json'


def apply_reviews(rows, annotation):
    indexed = {r['clip_id']: r for r in rows}
    reviews = annotation['reviews']
    if len({r['clip_id'] for r in reviews}) != len(reviews):
        raise ValueError('Duplicate review')
    for r in reviews:
        if r['clip_id'] not in indexed:
            raise ValueError('Unknown clip id')
        if r['reviewed_direction'] not in {'image_left', 'image_right', 'center', 'not_applicable'}:
            raise ValueError('Invalid reviewed direction')
        if r['display_direction'] != (r['reviewed_direction'] != 'not_applicable'):
            raise ValueError('Inconsistent direction display')
    by_id = {r['clip_id']: r for r in reviews}
    result = deepcopy(rows)
    for row in result:
        # Existing automatic fields, confidence, height and QA remain unchanged.
        row['automatic_dive_direction'] = row['dive_direction']
        row['final_dive_direction'] = row['dive_direction']
        row['display_dive_direction'] = row['dive_direction']
        row['final_direction_source'] = 'automatic_candidate'
        row['direction_user_reviewed'] = False
        if row['clip_id'] in by_id:
            r = deepcopy(by_id[row['clip_id']])
            row.update(final_dive_direction=r['reviewed_direction'],
                       display_dive_direction=r['reviewed_direction'] if r['display_direction'] else None,
                       final_direction_source=annotation['source'], direction_user_reviewed=True,
                       user_direction_review=r)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/goalkeeper_user_review_2026-10-11')
    args = parser.parse_args()
    source = BASE / 'summary.json'
    source_bytes = source.read_bytes()
    annotation = json.loads(REVIEWS.read_text(encoding='utf-8'))
    baseline = json.loads(source_bytes)
    rows = apply_reviews(baseline['rows'], annotation)
    out = args.output_dir
    if out.exists():
        raise FileExistsError('Choose a new output directory; prior exports are preserved')
    out.mkdir(parents=True)
    summary = dict(clips=len(rows), automatic_unknown=sum(r['dive_direction'] == 'unknown' for r in rows),
                   final_unknown=sum(r['final_dive_direction'] == 'unknown' for r in rows),
                   final_directions=dict(Counter(r['final_dive_direction'] for r in rows)),
                   user_reviewed_direction_records=len(annotation['reviews']),
                   automatic_accuracy_estimate=None, whole_record_QA_not_promoted=True,
                   keeper_own_side='not_camera_calibrated',
                   source_summary=str(source), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                   annotation=annotation, rows=rows)
    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    save('goalkeeper_display_results_39.json', summary)
    save('user_reviews.json', annotation)
    labels = {'image_right': '画面向右', 'not_applicable': '不展示方向（抗议）'}
    cells = ''.join(f'<tr><td>{html.escape(r["clip_id"])}</td><td>{labels[r["reviewed_direction"]]}</td>'
                    f'<td>{html.escape(r["note"])}</td></tr>' for r in annotation['reviews'])
    (out / 'report.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>门将方向：用户复核补充</title><style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:32px auto;padding:20px}}td,th{{padding:12px;border:1px solid #ccc}}table{{border-collapse:collapse}}</style>
<h1>门将方向：用户复核补充</h1><p>来源：2026-10-11 用户观看三条视频后在聊天中明确反馈。</p>
<table><tr><th>视频</th><th>展示结果</th><th>复核说明</th></tr>{cells}</table>
<p>自动结果仍是36条方向候选、3条unknown，原SAM2发布未修改。展示层为36条自动候选+2条用户方向标签+1条用户要求不展示方向；展示层未处理unknown为0。这不等于自动识别率或准确率100%。</p>
<p>第一条不编码成居中扑救。后两条“小范围”仅为用户定性描述，未新增位移测量或高度标签。方向统一采用画面坐标；“球员看向左”保留于原话，不作为相机标定结果。</p>
<p>本次只复核动作/方向，不验证其余36例、人物身份、扑救高度、进球结果或整条记录QA。</p>
<p><a href="goalkeeper_display_results_39.json">39条展示结果及原自动字段</a> · <a href="user_reviews.json">用户复核来源</a></p></html>''', encoding='utf-8')
    (out / 'README.txt').write_text('用户明确复核的方向补充，不覆盖自动识别。\n'
        '复现：python -X utf8 scripts/apply_goalkeeper_user_review.py --output-dir outputs/new_review_export\n'
        '展示请读取final_dive_direction和display_dive_direction（null表示不展示），不要读取原dive_direction字段冒充人工结果。\n'
        'direction_user_reviewed仅指方向/动作；不提升原QA和height，不意味着整条记录已通过。\n'
        '依赖仅Python标准库。本次未训练模型、未调阈值、未声称自动准确率提升。\n', encoding='utf-8')
    assert source.read_bytes() == source_bytes
    assert len(rows) == 39 and summary['automatic_unknown'] == 3 and summary['final_unknown'] == 0
    print(json.dumps({k: v for k, v in summary.items() if k not in {'rows', 'annotation'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
