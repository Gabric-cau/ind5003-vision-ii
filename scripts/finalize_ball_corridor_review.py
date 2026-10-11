"""Publish explicitly AI-only pixel notes and package integrity checks."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / '.deps_ball_mask'))
from scripts.run_ball_direction_corridor_review import load, save, sha, report, PARENT


NOTES = {
'col_eng_p01_falcao_col9': '第53、54帧新增选择落在可见球的模糊轨迹上；第58帧两种选择都覆盖球，主要是候选来源/中心变化。',
'col_eng_p02_kane_eng9': '第76帧仍覆盖可见球；第83、84帧球网附近的候选被停止接入，原裁剪中心未见清晰球体，无法据此确认结果。',
'col_eng_p03_cuadrado_col11': '第106、108、110帧仍覆盖模糊球，中心有变化。第116帧停止跟网格，第117帧新点仍落在球网背景，尾部误跟没有完全解决。',
'col_eng_p04_rashford_eng19': '第135、136帧新旧候选均覆盖可见的模糊球，主要是候选来源变化，不能称为新增识别。',
'col_eng_p06_henderson_eng8': '第127帧原候选落在背景人员附近，新关联没有继续接入；该帧属于短观察段。',
'col_eng_p07_uribe_col15': '混合结果：第160帧衣物误接被去掉，第164帧补上可见球；但第161帧旧选择能看到真球，新选择却落在草地/白线附近的差分残影，第161帧发生退步。第170、171帧网格点被去掉，第173帧仍误接网格。窄通道无法纠正错误的初始飞行方向。',
'col_eng_p08_trippier_eng12': '离脚附近观测从第80帧改到81帧，第84帧补上飞行球；但第87帧原裁剪可见真球，新版本漏选这一帧，仍有回收不足。',
'col_eng_p09_bacca_col7': '第130、132帧新旧选择都覆盖模糊球，主要是检测与差分候选的替换。',
'col_eng_p10_dier_eng4': '第124、125、127帧新旧裁剪仍覆盖球，候选中心有所变化；遮挡与守门员交叠阶段不能仅凭模型分数确认球身份。',
'cro_den_p04_kramaric_cro9': '第80帧两种选择都覆盖白线附近的模糊球，主要是候选来源变化。',
'cro_den_p07_schone_den19': '第76帧仍覆盖来球；第80至84帧不再持续接入守门员/手套附近候选。这些是短观察阶段，球是否被遮挡或接触仍未确认。',
'cro_den_p08_pivaric_cro22': '第158、159、160、161、163帧不再接入守门员躯干附近候选；原裁剪中心主要是黄色衣物，短观察阶段的身份切换减少。',
'esp_rus_p02_smolov_rus10': '第119帧新旧选择均覆盖可见球，主要是候选来源变化。',
'esp_rus_p03_pique_esp3': '第84、89帧仍覆盖可见球；第87帧原裁剪有清楚的飞行球，新版本漏选这一帧，属于需要修正的退步。',
'esp_rus_p05_koke_esp8': '第81、85帧新旧候选都覆盖可见球；本次变化不能当作准确率提升。',
'esp_rus_p06_golovin_rus17': '第98帧新旧选择均覆盖模糊球，主要是候选来源变化。',
'esp_rus_p08_cheryshev_rus6': '第75帧新旧选择均覆盖模糊球，主要是候选来源变化。',
'esp_rus_p09_aspas_esp17': '第104、109帧新增选择能看到飞行球；第106、107、110帧新旧选择都覆盖球，第109帧补回了此前的空缺。守门员附近短观察仍需核对，不能因此判定进球。',
'rus_cro_p03_dzagoev_rus9': '第96、97帧新旧目标都落在守门员腿/球鞋附近，候选ID变化没有修正这段短观察的身份误跟。',
'rus_cro_p04_kovacic_cro8': '第92、93帧不再接入守门员手臂/球网附近候选。该阶段存在遮挡，不能将停止关联等同于没进球。',
'rus_cro_p06_modric_cro10': '第155帧原裁剪可见飞行球，新版本漏选；第156帧新旧仍覆盖球。第162、163帧新点转到守门员身上，原点在网内白色目标附近，短观察存在明显退步。',
'rus_cro_p07_ignashevich_rus4': '第70帧新增神经候选能看到飞行球，补回此前的空缺。',
'rus_cro_p09_kuzyaev_rus7': '第82帧仍接在守门员附近，第83帧又接到裁判/守门员附近，第84帧停止接入。短观察的衣物身份误跟仍未解决。',
'rus_cro_p10_rakitic_cro7': '第88至90帧停止接入守门员/球鞋附近候选；第93帧新点落在球网背景，未见清晰球体，末端身份仍不能确认。',
'esp_rus_p01_iniesta_esp6': '新旧短窗口选择一致，第67帧和观察范围保留。但是第67帧当前所选点在球网/广告背景附近，裁剪未见清楚球体；保留候选和帧不能称为修复了用户指出的真球跟踪。第59帧裁剪可见来球，第62帧已出现球门网格混淆。'
}


class References(HTMLParser):
    def __init__(self): super().__init__(); self.refs = []
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('src', 'href') and value and not value.startswith(('#', 'http:', 'https:', 'data:')):
                self.refs.append(value.split('#')[0])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); out = args.directory.resolve()
    summary = load(out / 'summary.json'); protocol = load(out / 'protocol.json'); validation = load(out / 'validation.json')
    evidence = {q['clip_id']: q for q in load(out / 'changed_pixel_evidence.json')['rows']}
    assert summary['processed_clips'] == validation['association_clips'] == validation['rendered_clips'] == 39
    assert len(summary['rows']) == len({q['clip_id'] for q in summary['rows']}) == 39
    assert all(q['media'] for q in summary['rows'])
    assert load(out / 'tests.json')['successful'] is True
    for rel, digest in protocol['code_sha256'].items():
        assert sha(ROOT / rel) == sha(out / 'source_snapshot' / Path(rel).name) == digest
    reviewed = []
    for cid, note in NOTES.items():
        item = evidence.get(cid)
        reviewed.append({'clip_id': cid, 'reviewer': 'assistant_visual_inspection', 'human_verified': False,
            'accuracy_validated': False, 'note': note,
            'evidence_frames': [q['frame'] for q in item['frames']] if item else [44, 59, 62, 67, 68],
            'panel': f'media/{item["panel"]}' if item else 'media/iniesta_67_retained.jpg'})
    save(out / 'visual_review.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer': 'assistant_visual_inspection',
        'scope': 'all changed-frame original-pixel panels in 24 clips; extra Iniesta pixel crop and two rendered sheets',
        'reviewed_clips': len(reviewed), 'human_verified': False, 'accuracy_validated': False,
        'used_for_association': False, 'assessment': 'mixed_results_not_a_validated_accuracy_upgrade', 'rows': reviewed})
    report(out, summary['rows'])
    doc = (out / 'report.html').read_text(encoding='utf-8')
    banner = '<p class="note">AI原像素复核发现有收益也有退步：部分守门员/背景候选减少，但Uribe第161帧仍选残影，Trippier第87帧、Piqué第87帧、Modrić第155帧真球漏选。Iniesta第67帧虽保留，所选点仍有球网混淆。当前版本作为实验对照，不能宣称整体准确率提升。</p>'
    doc = doc.replace('<h1>球：窄通道连续关联</h1>', '<h1>球：窄通道连续关联</h1>' + banner)
    for row in reviewed:
        cid = row['clip_id']
        needle = f'<p><a href="results/{cid}.json">'
        doc = doc.replace(needle, f'<p><a href="{row["panel"]}">本轮原像素对照</a></p>' + needle)
    (out / 'report.html').write_text(doc, encoding='utf-8')
    parser = References(); parser.feed(doc)
    missing = [q for q in parser.refs if not (out / q).resolve().is_file()]
    assert not missing, missing
    all_main_same = all(q['main_end'] == q['control_main_end'] for q in summary['rows'])
    assert all_main_same
    retained = load(out / 'results/esp_rus_p01_iniesta_esp6.json')
    assert any(q['frame'] == 67 for q in retained['trajectory_in_focus'])
    continuity = []
    for row in summary['rows']:
        inp = load(PARENT / 'inputs' / f'{row["clip_id"]}.json')
        keys = sorted(int(f) for f in inp['candidates'])
        assert keys == list(range(keys[0], keys[-1] + 1))
        r = load(out / 'results' / f'{row["clip_id"]}.json')
        assert sha(PARENT / 'inputs' / f'{row["clip_id"]}.json') == r['parent_inputs_sha256']
        assert sha(PARENT / 'results' / f'{row["clip_id"]}.json') == r['parent_result_sha256']
        assert r['outcome'] == 'unknown'
        continuity.append(row['clip_id'])
    save(out / 'delivery_check.json', {'processed_clips': 39, 'decoded_videos': 39,
        'decoded_video_frames': validation['decoded_video_frames'], 'local_references_checked': len(parser.refs),
        'missing_references': [], 'main_end_matches_same_window_control_for_all_clips': all_main_same,
        'changed_clips': sum(bool(q['changed_source_frames']) for q in summary['rows']),
        'inputs_have_consecutive_source_frame_keys': True, 'frozen_parent_files_unchanged': True,
        'iniesta_frame_67_included': True, 'iniesta_frame_67_identity_verified': False,
        'tests_run': load(out / 'tests.json')['tests_run'], 'code_hashes_match': True,
        'visual_reviewed_clips': len(reviewed), 'browser_interaction_verified': False,
        'human_verified': False, 'accuracy_validated': False, 'outcome_counts': {'unknown': 39},
        'delivery_status': 'experimental_comparison_with_recorded_regressions'})
    (out / 'source_snapshot' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(out / 'delivery_protocol.json', {'code_sha256': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
         'visual_notes_provenance': 'assistant inspection of displayed original-pixel panels, not human ground truth'})
    print(load(out / 'delivery_check.json'))


if __name__ == '__main__': main()
