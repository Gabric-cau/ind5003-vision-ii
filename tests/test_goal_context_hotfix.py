import json
from pathlib import Path

import pytest

from scripts import apply_goal_context_hotfix as hotfix


@pytest.fixture
def bundle_fixture(tmp_path, monkeypatch):
    bundle, payload = tmp_path / "bundle", tmp_path / "payload"
    bundle.mkdir()
    payload.mkdir()

    def write(root, name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return hotfix.digest(data)

    old_hashes, new_hashes = {}, {}
    for name, allowed in hotfix.ALLOWED_FUNCTIONS.items():
        prefix = "import math\n\ndef unchanged_helper(x):\n    return x + 1\n\n"
        before = prefix + "\n\n".join(f"def {function}(x):\n    return x" for function in sorted(allowed)) + "\n"
        after = prefix + "\n\n".join(f"def {function}(x):\n    return x + 2" for function in sorted(allowed)) + "\n"
        old_hashes[name] = write(bundle, name, before.encode())
        new_hashes[name] = write(payload, name, after.encode())
    helper = write(bundle, "src/contact_video_features.py", b"from src.step_pipeline import unchanged_helper\n")
    weights = write(bundle, "models/contact/deployment_weights.npz", b"frozen contact weights")
    visual = write(bundle, "models/detector.pt", b"frozen detector weights")
    metadata = {"extractor_code_sha256": {"src/step_pipeline.py": old_hashes["src/step_pipeline.py"],
                                          "src/contact_video_features.py": helper},
                "weights_sha256": weights, "visual_model_sha256": {"detector.pt": visual},
                "feature_names": ["original_feature"]}
    old_hashes[hotfix.METADATA] = write(bundle, hotfix.METADATA, hotfix.encode_json(metadata))
    records = [{"path": p.relative_to(bundle).as_posix(), "bytes": p.stat().st_size,
                "sha256": hotfix.file_digest(p)} for p in sorted(bundle.rglob("*")) if p.is_file()]
    manifest = {"schema_version": "offline-video-portable-bundle-v1", "files": records,
                "file_count_excluding_manifest_and_checksum_list": len(records),
                "payload_bytes": sum(row["bytes"] for row in records), "frozen_snapshot": "test-v5"}
    old_hashes[hotfix.MANIFEST] = write(bundle, hotfix.MANIFEST, hotfix.encode_json(manifest))
    sums = {row["path"]: row["sha256"] for row in records}
    sums[hotfix.MANIFEST] = old_hashes[hotfix.MANIFEST]
    old_hashes[hotfix.CHECKSUMS] = write(bundle, hotfix.CHECKSUMS,
        ("\n".join(f"{sums[name]}  {name}" for name in sorted(sums)) + "\n").encode())
    monkeypatch.setattr(hotfix, "ORIGINAL_HASHES", old_hashes)
    monkeypatch.setattr(hotfix, "PATCHED_HASHES", new_hashes)
    return bundle, payload


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_apply_preserves_weights_and_exact_backup_and_updates_bindings(bundle_fixture):
    bundle, payload = bundle_fixture
    before = snapshot(bundle)
    result = hotfix.apply_hotfix(bundle, payload)
    assert result["status"] == "applied"
    assert result["weights_changed"] is False
    assert result["accuracy_improvement_established"] is False
    for name, data in before.items():
        if name in hotfix.ORIGINAL_HASHES:
            assert (bundle / hotfix.BACKUP_NAME / name).read_bytes() == data
        else:
            assert (bundle / name).read_bytes() == data
    metadata = json.loads((bundle / hotfix.METADATA).read_bytes())
    assert metadata["extractor_code_sha256"]["src/step_pipeline.py"] == hotfix.PATCHED_HASHES["src/step_pipeline.py"]
    assert metadata["feature_names"] == ["original_feature"]
    audit = metadata["compatibility_updates"][-1]
    assert audit["original_metadata_sha256"] == hotfix.digest(before[hotfix.METADATA])
    assert audit["contact_weights_and_visual_models_unchanged"] is True
    manifest = json.loads((bundle / hotfix.MANIFEST).read_bytes())
    for row in manifest["files"]:
        data = (bundle / row["path"]).read_bytes()
        assert row["sha256"] == hotfix.digest(data)
        assert row["bytes"] == len(data)
    assert manifest["payload_bytes"] == sum(row["bytes"] for row in manifest["files"])
    sums = hotfix.parse_checksums((bundle / hotfix.CHECKSUMS).read_bytes())
    assert len(sums) == len(manifest["files"]) + 1
    for name, expected in sums.items():
        assert hotfix.file_digest(bundle / name) == expected


def test_check_only_has_no_side_effects(bundle_fixture):
    bundle, payload = bundle_fixture
    before = snapshot(bundle)
    assert hotfix.apply_hotfix(bundle, payload, check_only=True)["status"] == "ready"
    assert snapshot(bundle) == before
    assert not (bundle / hotfix.BACKUP_NAME).exists()


@pytest.mark.parametrize("name", ["src/step_pipeline.py", hotfix.METADATA, hotfix.MANIFEST,
                                 hotfix.CHECKSUMS, "models/contact/deployment_weights.npz"])
def test_modified_original_refused_without_mutation(bundle_fixture, name):
    bundle, payload = bundle_fixture
    path = bundle / name
    path.write_bytes(path.read_bytes() + b"changed")
    before = snapshot(bundle)
    with pytest.raises(ValueError):
        hotfix.apply_hotfix(bundle, payload)
    assert snapshot(bundle) == before
    assert not (bundle / hotfix.BACKUP_NAME).exists()


def test_unpinned_payload_refused_without_mutation(bundle_fixture):
    bundle, payload = bundle_fixture
    path = payload / "src/fresh_event_analysis.py"
    path.write_bytes(path.read_bytes() + b"# not the reviewed patch\n")
    before = snapshot(bundle)
    with pytest.raises(ValueError, match="Unrecognized hotfix payload"):
        hotfix.apply_hotfix(bundle, payload)
    assert snapshot(bundle) == before


@pytest.mark.parametrize("change", ["helper", "signature", "import", "missing"])
def test_ast_guard_rejects_unrelated_changes_even_with_matching_payload_pin(bundle_fixture, monkeypatch, change):
    bundle, payload = bundle_fixture
    path = payload / "src/step_pipeline.py"
    data = path.read_text()
    if change == "helper":
        data = data.replace("return x + 1", "return x + 99")
    elif change == "signature":
        data = data.replace("def process_video_steps(x)", "def process_video_steps(x, y)")
    elif change == "import":
        data = data.replace("import math", "import os")
    else:
        data = data.replace("def process_video_steps(x)", "def another_function(x)")
    path.write_text(data)
    monkeypatch.setitem(hotfix.PATCHED_HASHES, "src/step_pipeline.py", hotfix.file_digest(path))
    before = snapshot(bundle)
    with pytest.raises(ValueError, match="AST|signature|Missing hotfix"):
        hotfix.apply_hotfix(bundle, payload)
    assert snapshot(bundle) == before
    assert not (bundle / hotfix.BACKUP_NAME).exists()


def test_bad_contact_binding_refused_even_if_file_pins_updated(bundle_fixture, monkeypatch):
    bundle, payload = bundle_fixture
    path = bundle / hotfix.METADATA
    metadata = json.loads(path.read_bytes())
    metadata["extractor_code_sha256"]["src/step_pipeline.py"] = "0" * 64
    path.write_bytes(hotfix.encode_json(metadata))
    manifest_path = bundle / hotfix.MANIFEST
    manifest = json.loads(manifest_path.read_bytes())
    for row in manifest["files"]:
        if row["path"] == hotfix.METADATA:
            row.update(bytes=path.stat().st_size, sha256=hotfix.file_digest(path))
    manifest["payload_bytes"] = sum(row["bytes"] for row in manifest["files"])
    manifest_path.write_bytes(hotfix.encode_json(manifest))
    sums = {row["path"]: row["sha256"] for row in manifest["files"]}
    sums[hotfix.MANIFEST] = hotfix.file_digest(manifest_path)
    (bundle / hotfix.CHECKSUMS).write_text("\n".join(f"{sums[name]}  {name}" for name in sorted(sums)) + "\n")
    for name in (hotfix.METADATA, hotfix.MANIFEST, hotfix.CHECKSUMS):
        monkeypatch.setitem(hotfix.ORIGINAL_HASHES, name, hotfix.file_digest(bundle / name))
    before = snapshot(bundle)
    with pytest.raises(ValueError, match="different step-pipeline binding"):
        hotfix.apply_hotfix(bundle, payload)
    assert snapshot(bundle) == before


def test_second_apply_keeps_first_backup(bundle_fixture):
    bundle, payload = bundle_fixture
    hotfix.apply_hotfix(bundle, payload)
    before = snapshot(bundle)
    with pytest.raises(FileExistsError, match="backup already exists"):
        hotfix.apply_hotfix(bundle, payload)
    assert snapshot(bundle) == before


def test_write_failure_restores_originals(bundle_fixture, monkeypatch):
    bundle, payload = bundle_fixture
    before = snapshot(bundle)
    original_replace = hotfix.atomic_replace
    calls = 0

    def fail_once(path, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated disk failure")
        original_replace(path, data)

    monkeypatch.setattr(hotfix, "atomic_replace", fail_once)
    with pytest.raises(OSError, match="simulated disk failure"):
        hotfix.apply_hotfix(bundle, payload)
    for name, data in before.items():
        assert (bundle / name).read_bytes() == data
    for name in hotfix.ORIGINAL_HASHES:
        assert (bundle / hotfix.BACKUP_NAME / name).read_bytes() == before[name]
