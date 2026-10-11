"""Read-only regression checks for a fresh camera run; never relabel source data.

Example: python scripts/retest_camera_timing.py --run-dir outputs/camera_timing_retest_2026-10-10_01
Run the camera pipeline in that NEW directory first. Results are written only there.
Unittest discovery does not collect standalone pytest functions.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import html
import io
import json
import os
import sys
import unittest

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import yaml
from src.camera_timing_v2 import playback_evidence

cv2.setNumThreads(2)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def tests(out):
    results = {}
    for label, patterns in (
        ('targeted', ('test_camera_timing_v2.py', 'test_goal_geometry.py',
                      'test_goal_plane_rectification.py', 'test_goal_event_tracker.py')),
        ('full_unittest_discovery', ('test*.py',)),
    ):
        loader = unittest.TestLoader()
        suite = unittest.TestSuite(loader.discover(str(ROOT / 'tests'), pattern=p) for p in patterns)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (out / (label + '.log')).write_text(stream.getvalue(), encoding='utf-8')
        results[label] = {
            'tests_run': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
                      - len(result.expectedFailures) - len(result.unexpectedSuccesses),
            'failures': [{'test': str(t), 'traceback': msg} for t, msg in result.failures],
            'errors': [{'test': str(t), 'traceback': msg} for t, msg in result.errors],
            'skipped': result.skipped, 'success': result.wasSuccessful(),
            'patterns': patterns,
        }
        print(json.dumps({'stage': label, 'run': result.testsRun, 'errors': len(result.errors),
                          'failures': len(result.failures)}, ensure_ascii=False), flush=True)
    results['scope_warning'] = 'Unittest does not collect standalone pytest functions. Passing tests are not real-video accuracy.'
    save(out / 'retest_tests.json', results)
    return results


def diagnose_pair(reference, query):
    """Expose the FIRST failing production gate without relaxing its thresholds."""
    a = np.asarray([r['feature'] for r in reference], float)
    b = np.asarray([r['feature'] for r in query], float)
    rt0 = [r['time_sec'] for r in reference]
    qt0 = [r['time_sec'] for r in query]
    production = playback_evidence(rt0, a, qt0, b)
    result = {'reference_valid_samples': len(a), 'query_valid_samples': len(b),
              'production_status': production['status'], 'production_reason': production['reason']}

    def stop(gate):
        return {**result, 'first_failing_gate': gate}

    if len(a) < 10 or len(b) < 10:
        return stop('fewer_than_10_valid_samples')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        return stop('feature_shape')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return stop('nonfinite_feature')
    result.update(reference_variation=float(np.std(a, axis=0).mean()),
                  query_variation=float(np.std(b, axis=0).mean()))
    if min(result['reference_variation'], result['query_variation']) < .06:
        return stop('insufficient_action_variation')
    dist = np.sqrt(np.mean((b[:, None, :] - a[None, :, :]) ** 2, axis=2))
    index = dist.argmin(axis=1)
    good = dist[np.arange(len(b)), index] < .16
    result['matched_samples'] = int(good.sum())
    if good.sum() < 10:
        return stop('fewer_than_10_pose_matches')
    rt = np.asarray(rt0)[index[good]]
    qt = np.asarray(qt0)[good]
    result.update(ordered_fraction=float(np.mean(np.diff(rt) >= -.10)),
                  reference_span_sec=float(np.ptp(rt)), query_span_sec=float(np.ptp(qt)))
    if result['ordered_fraction'] < .85:
        return stop('nonmonotonic_pose_matches')
    if np.ptp(rt) < .5 or np.ptp(qt) < .8:
        return stop('insufficient_temporal_span')
    slope, intercept = np.polyfit(rt, qt, 1)
    residual = float(np.median(np.abs(qt - (slope * rt + intercept))))
    result.update(relative_time_stretch=float(slope), median_residual_sec=residual)
    if slope <= 0 or residual > .22:
        return stop('nonpositive_stretch_or_large_residual')
    return stop(None)


def compare_and_decode(baseline, out):
    old = read(baseline / 'camera_timings_candidates.json')
    new = read(out / 'camera_timings_candidates.json')
    manifest = yaml.safe_load((ROOT / 'configs/penalty_slicing_manifest.yaml').read_text(encoding='utf-8'))
    ids = [f"{m}_p{p['number']:02d}_{p['kicker']}" for m, v in manifest.items() for p in v['penalties']]
    assert [c['clip_id'] for c in old['clips']] == ids
    assert [c['clip_id'] for c in new['clips']] == ids
    snapshots = {p.name: sha(p) for p in (baseline / 'camera_timings_candidates.json', baseline / 'summary.json')}
    previous_media = {r['clip_id']: r for r in read(baseline / 'media_audit.json')['records']}
    media, differences, pairs, reused = [], [], [], []
    contiguous = True
    for old_clip, clip in zip(old['clips'], new['clips'], strict=True):
        cid = clip['clip_id']
        if clip != old_clip:
            differences.append({'clip_id': cid, 'changed_top_level_keys': [k for k in set(old_clip) | set(clip)
                                if old_clip.get(k) != clip.get(k)]})
        last = 0
        for segment in clip['segments']:
            contiguous &= segment['start_frame'] == last and segment['end_frame_exclusive'] > last
            last = segment['end_frame_exclusive']
        contiguous &= last == clip['frame_count']
        source = ROOT / 'data/raw_clips' / (cid + '.mp4')
        video = baseline / 'review' / clip['video']
        cap = cv2.VideoCapture(str(video))
        decoded = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.read()[0]:
            decoded += 1
        cap.release()
        media.append({'clip_id': cid, 'decoded_frames': decoded, 'expected_frames': clip['frame_count'],
                      'fps': fps, 'decode_pass': decoded == clip['frame_count'] and abs(fps - clip['fps']) < 1e-6,
                      'original_sha_matches_baseline_and_fresh': sha(source) == old_clip['source_sha256'] == clip['source_sha256'],
                      'review_media_unchanged': sha(video) == previous_media[cid]['output_sha256']})
        action = read(out / 'action_v3' / (cid + '.json'))
        if action.get('reused_sequence_segment_ids'):
            reused.append(cid)
        for sid in action['evidence']:
            ref = action['sequences'][str(action['reference_segment_id'])]
            for role in ('kicker', 'keeper'):
                detail = diagnose_pair(ref[role], action['sequences'][sid][role])
                expected = action['evidence'][sid]['role_results'][role]
                assert detail['production_status'] == expected['status']
                assert detail['production_reason'] == expected['reason']
                pairs.append({'clip_id': cid, 'segment_id': int(sid), 'role': role, **detail})
        print(json.dumps({'stage': 'decode_and_compare', 'clip': cid, 'decoded': decoded}, ensure_ascii=False), flush=True)
    result = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable,
        'baseline': str(baseline), 'fresh_run': str(out), 'manifest_clips': len(ids),
        'candidate_json_identical': old == new,
        'candidate_csv_identical': sha(baseline / 'camera_timings_candidates.csv') == sha(out / 'camera_timings_candidates.csv'),
        'clip_differences': differences, 'contiguous_full_timeline': bool(contiguous),
        'fresh_summary': new['summary'], 'reused_action_sequence_clips': reused,
        'baseline_files_unchanged_during_audit': all(sha(baseline / n) == s for n, s in snapshots.items()),
        'classifier_unchanged_from_baseline': sha(ROOT / 'src/camera_timing_v2.py') == old['classifier_sha256'],
        'media': media, 'media_decode_pass_count': sum(r['decode_pass'] for r in media),
        'original_unchanged_count': sum(r['original_sha_matches_baseline_and_fresh'] for r in media),
        'review_media_unchanged_count': sum(r['review_media_unchanged'] for r in media),
        'slow_motion': {'tested_segments': len(pairs) // 2, 'role_pairs': len(pairs),
                        'first_failure_counts': dict(Counter(p['first_failing_gate'] for p in pairs)), 'details': pairs},
        'human_verified_segments': 0, 'class_accuracy': None, 'boundary_accuracy': None,
        'limitations': [
            'Reproducibility and video decoding are not semantic accuracy.',
            'No human review or independent labels were created by this retest.',
            'Fewer than 10 valid samples identifies an upstream bottleneck, not a unique cause. Identity/visibility gates require separate instrumentation.',
            'Fresh review HTML lacks its own media copies; browser tests use the unchanged baseline review server.',
            'Coverage includes unknown classes; not_detected is not verified absence.',
        ],
    }
    save(out / 'retest_regression.json', result)
    return result


def report(out):
    result = read(out / 'retest_regression.json')
    result['tests'] = read(out / 'retest_tests.json')
    if (out / 'retest_browser.json').exists():
        result['browser'] = read(out / 'retest_browser.json')
    save(out / 'retest_report.json', result)
    summary = result['fresh_summary']
    targeted = result['tests']['targeted']
    full = result['tests']['full_unittest_discovery']
    gate_names = {
        'fewer_than_10_valid_samples': '参考段或待测段的有效姿态样本不足 10 个',
        'nonmonotonic_pose_matches': '跨镜头匹配的动作顺序不一致',
        'insufficient_action_variation': '动作变化幅度不足',
    }
    gates = {gate_names.get(k, k): v for k, v in result['slow_motion']['first_failure_counts'].items()}
    classes = summary['task_class_candidates']
    class_names = {'back_of_kicker': '罚球者背后', 'angle_45_right': '罚球者右侧斜机位',
                   'slowmo_kicker_closeup': '慢动作罚球者特写', 'slowmo_gk_back': '慢动作守门员背后'}
    class_text = '；'.join(f'{name} {classes.get(key, 0)} 段' for key, name in class_names.items())
    falcao = next((p for p in result['slow_motion']['details'] if p['clip_id'] == 'col_eng_p01_falcao_col9'
                   and p['segment_id'] == 3 and p['role'] == 'kicker'), None)
    example = '' if falcao is None else ('<p>具体示例：Falcao 的第 4 个镜头（segment_id=3），参考段有效姿态 '
        f"{falcao['reference_valid_samples']} 个，待测段 {falcao['query_valid_samples']} 个；首个未通过条件："
        + html.escape(str(gate_names.get(falcao['first_failing_gate'], falcao['first_failing_gate']))) + '。</p>')
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Vision II 机位模块复测</title>
<style>body{{max-width:1000px;margin:40px auto;font:16px/1.65 system-ui;padding:0 20px;color:#173042}}pre{{white-space:pre-wrap;background:#f1f5f7;padding:18px;border-radius:8px}}h1{{font-size:28px}}</style>
<h1>Vision II 机位时间识别：独立复测</h1>
<p>本报告记录软件回归与可复现性，不代表模型准确率。未修改算法阈值，未生成任何人工标签。</p>
<ul><li>原始视频：{result['manifest_clips']} 条；候选镜头：{summary['segments']} 个；时间轴连续覆盖：{result['contiguous_full_timeline']}。</li>
<li>新旧候选 JSON 完全一致：{result['candidate_json_identical']}；CSV 完全一致：{result['candidate_csv_identical']}。</li>
<li>原始视频哈希一致：{result['original_unchanged_count']}；复核副本重新完整解码通过：{result['media_decode_pass_count']}。</li>
<li>定向单元测试：{targeted['passed']}/{targeted['tests_run']} 通过；全项目 unittest：{full['passed']} 通过，{len(full['errors'])} 个错误，{len(full['failures'])} 个断言失败。unittest 不收集普通 pytest 函数。</li>
<li>慢动作：{result['slow_motion']['tested_segments']} 个片段进入对齐，{result['slow_motion']['role_pairs']} 组角色对齐；结果仍以详细记录为准。</li>
<li>人工验证镜头：0；机位类别准确率和边界准确率：均未测得。</li></ul>
<h2>对作业要求的含义</h2>
<p>当前仍是候选结果：{class_text}。候选数不代表已验证完成度。</p>
<p>{summary['segments']} 段中有 {result['slow_motion']['tested_segments']} 段进入慢动作对齐，每段分别尝试罚球者和守门员，共 {result['slow_motion']['role_pairs']} 组。其他片段未进入该对齐流程，不应把角色对齐数当成视频数或准确率分母。</p>
<h2>慢动作的首个未通过条件</h2><pre>{html.escape(json.dumps(gates, ensure_ascii=False, indent=2))}</pre>
<p>完整关节点与唯一球衣匹配是当前动作特征的前提；有效样本不足不能单独证明是遮挡、身份匹配还是姿态漏检。当前六关节角特征也不具备跨视角不变性。不能据此宣称视频本身不清晰或没有慢动作。</p>
{example}
<p>后续可实验：允许部分关节点参与匹配、结合时序身份追踪、改用有顺序约束的动作对齐。以上是待验证方向，本次复测没有实施，也没有通过放宽阈值让结果“过关”。</p>
<h2>浏览器测试</h2><pre>{html.escape(json.dumps(result.get('browser', {'status':'not yet recorded'}), ensure_ascii=False, indent=2))}</pre>
<h2>全项目测试错误</h2><pre>{html.escape(json.dumps(full['errors'], ensure_ascii=False, indent=2))}</pre>
<p>机器可读明细：<a href="retest_report.json">retest_report.json</a>；本报告不覆盖射门脚、步数、球轨迹或进球判断模块的真实准确率。</p>
</html>'''
    (out / 'retest_report.html').write_text(body, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/camera_timing_2026-10-10/release')
    parser.add_argument('--stage', choices=('tests', 'compare', 'report', 'all'), default='all')
    args = parser.parse_args()
    out, baseline = args.run_dir.resolve(), args.baseline.resolve()
    if not out.is_dir() or out == baseline or baseline.is_relative_to(out) or out.is_relative_to(baseline):
        parser.error('Use a separate existing fresh-run directory, not the baseline or its parent/child.')
    if args.stage in ('tests', 'all'):
        tests(out)
    if args.stage in ('compare', 'all'):
        compare_and_decode(baseline, out)
    if args.stage in ('report', 'all'):
        report(out)
