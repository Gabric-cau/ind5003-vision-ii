"""Explicitly scoped synthetic and replay checks for the evening experiment."""
from pathlib import Path
import sys,json,unittest,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.validate_penalty_phase_experiment import tests as historical_tests
from scripts.validate_terminal_footwork import check_result
from src.style_aware_steps import analyze_style_aware_steps
from scripts.run_terminal_footwork import sha
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
history=historical_tests();suite=unittest.TestSuite();counts={}
for name in ("test_terminal_footwork","test_style_aware_steps","test_contact_transition","test_goalkeeper_bounded","test_ball_expired_paths","test_goal_geometry","test_person_appearance","test_goalkeeper_qa_gap","test_goalkeeper_observation_guard"):
    spec=importlib.util.spec_from_file_location(name,ROOT/"tests"/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sub=unittest.defaultTestLoader.loadTestsFromModule(module);counts[name]=sub.countTestCases();suite.addTests(sub)
outcome=unittest.TextTestRunner(verbosity=1).run(suite)
errors=[];paths=[p for p in (RUN/"steps_v4/classified").glob("*.json") if p.name!="summary.json"]
if len(paths)!=39:errors.append("expected 39 new step results")
for path in paths:
    r=json.loads(path.read_text());errors.extend(f"{path.stem}: {e}" for e in check_result(r))
    if analyze_style_aware_steps(r)!=r["step_analysis"]:errors.append(f"{path.stem}: replay mismatch")
    if r["step_analysis"]["verified_step_count"] is not None:errors.append(f"{path.stem}: fabricated verified count")
    c=r["contact"];a,b=c["interval_frames"]
    if not r["shot_start_frame"]<=c["contact_frame"]<r["shot_end_frame_exclusive"]:errors.append(f"{path.stem}: contact out of shot")
media=json.loads((ROOT/"outputs/phase_optimization_2026-10-09/review/steps/media_validation.json").read_text())
for row in media["reports"]:
    if sha(Path(row["source"]))!=row["source_sha256"]:errors.append("raw video hash changed: "+row["source"])
report={"passed":not errors and history["passed"] and outcome.wasSuccessful(),"errors":errors,
        "historical_tests":history,"additional_tests":counts,"total_tests":history["tests_run"]+outcome.testsRun,
        "new_result_replays":len(paths),"raw_video_hashes_checked":len(media["reports"]),
        "human_verified":False,"accuracy_validated":False,"full_pytest_suite_run":False,
        "scope":"Selected synthetic tests, count partitions, hashes and deterministic replay; NOT real-video accuracy"}
(RUN/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2));sys.exit(0 if report["passed"] else 1)
