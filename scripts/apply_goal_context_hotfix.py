"""Apply the reviewed goal-context hotfix to the exact portable v5 bundle.

This standalone utility uses only the Python standard library. Place it beside
the payload's src directory, or pass --payload-dir explicitly.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


HOTFIX_ID = "goal-context-analysis-only-2026-10-30"
BACKUP_NAME = "_goal_context_hotfix_backup_2026-10-30"
METADATA = "models/contact/deployment_metadata.json"
MANIFEST = "bundle_manifest.json"
CHECKSUMS = "SHA256SUMS"
ALLOWED_FUNCTIONS = {
    "src/step_pipeline.py": {"_shot_goal_context", "process_video_steps"},
    "src/fresh_event_analysis.py": {"contact_window"},
}
ORIGINAL_HASHES = {
    "src/step_pipeline.py": "866817be7de157b8455c2948532ced9b5ef04021ed69721863f01043ccaddcb4",
    "src/fresh_event_analysis.py": "b90bd0b7a271e2295027eaa96aabe23c3899683afa06cca3cd942a1ec58d9837",
    METADATA: "098157095abf40d6820305be961683a85e1770b78a73425b5cd783441cc6674b",
    MANIFEST: "432abfaa2e2740892cb2e7f474d57921f2f94960bbbe946e3cb5744c0ef8d797",
    CHECKSUMS: "b2d1e748ed69d19bdd4dabe54ae62b50000e5173682aad4dd2fe012f0d679f58",
}
PATCHED_HASHES = {
    "src/step_pipeline.py": "9cf9f123c96d6cc2c3d3b213aca41ad532846beecca0e2aaf63de321757ea15d",
    "src/fresh_event_analysis.py": "940525f2a6ba62747a4a652c9074bb4c21c41e725bdd45367caae04f4be73af3",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def bound_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if (not relative or parts.is_absolute() or ".." in parts.parts
            or "\\" in relative or str(parts) != relative):
        raise ValueError("Unsafe bundle path: " + relative)
    path = root
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink is not a frozen bundle file: " + relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes bundle: " + relative)
    return path


def encode_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def validate_ast_change(before: bytes, after: bytes, allowed: set[str], name: str) -> list[str]:
    """Permit body changes only in explicitly named top-level functions."""
    trees = [ast.parse(data.decode("utf-8"), filename=name) for data in (before, after)]
    stripped = []
    bodies = []
    for tree in trees:
        found = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed:
                if node.name in found:
                    raise ValueError("Duplicate hotfix function: " + node.name)
                found[node.name] = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
                node.body = []
        if set(found) != allowed:
            raise ValueError("Missing hotfix function in " + name)
        bodies.append(found)
        stripped.append(ast.dump(tree, include_attributes=False))
    if stripped[0] != stripped[1]:
        raise ValueError("Unrelated AST or function signature changed in " + name)
    changed = sorted(key for key in allowed if bodies[0][key] != bodies[1][key])
    if set(changed) != allowed:
        raise ValueError("Expected hotfix function bodies were not all changed in " + name)
    return changed


def parse_checksums(data: bytes) -> dict[str, str]:
    rows = {}
    for line in data.decode("utf-8").splitlines():
        checksum, separator, name = line.partition("  ")
        if (not separator or len(checksum) != 64 or name in rows
                or any(char not in "0123456789abcdef" for char in checksum)):
            raise ValueError("Invalid or duplicate SHA256SUMS entry")
        rows[name] = checksum
    return rows


def preflight(bundle: Path, payload: Path):
    bundle, payload = bundle.resolve(), payload.resolve()
    if not bundle.is_dir() or not payload.is_dir():
        raise ValueError("Bundle and payload directories must exist")
    if (bundle / BACKUP_NAME).exists():
        raise FileExistsError("Hotfix backup already exists; refusing to replace it")
    originals = {}
    for name, expected in ORIGINAL_HASHES.items():
        data = bound_path(bundle, name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Not the supported original v5 file: " + name)
        originals[name] = data
    replacements, ast_changes = {}, {}
    for name, expected in PATCHED_HASHES.items():
        data = bound_path(payload, name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Unrecognized hotfix payload: " + name)
        ast_changes[name] = validate_ast_change(originals[name], data, ALLOWED_FUNCTIONS[name], name)
        replacements[name] = data

    manifest = json.loads(originals[MANIFEST])
    if manifest.get("schema_version") != "offline-video-portable-bundle-v1":
        raise ValueError("Unsupported bundle manifest schema")
    records = {}
    for row in manifest["files"]:
        name = row["path"]
        if name in records or name in {MANIFEST, CHECKSUMS}:
            raise ValueError("Invalid or duplicate manifest entry: " + name)
        path = bound_path(bundle, name)
        if path.stat().st_size != row["bytes"] or file_digest(path) != row["sha256"]:
            raise ValueError("Frozen bundle file differs: " + name)
        records[name] = row
    if manifest["file_count_excluding_manifest_and_checksum_list"] != len(records):
        raise ValueError("Manifest file count differs")
    if manifest["payload_bytes"] != sum(row["bytes"] for row in records.values()):
        raise ValueError("Manifest payload size differs")
    if not set(replacements).union({METADATA}) <= records.keys():
        raise ValueError("Manifest omits a patched file")
    checksums = parse_checksums(originals[CHECKSUMS])
    expected_sums = {name: row["sha256"] for name, row in records.items()}
    expected_sums[MANIFEST] = digest(originals[MANIFEST])
    if checksums != expected_sums:
        raise ValueError("SHA256SUMS disagrees with bundle manifest")

    metadata = json.loads(originals[METADATA])
    if metadata["extractor_code_sha256"].get("src/step_pipeline.py") != ORIGINAL_HASHES["src/step_pipeline.py"]:
        raise ValueError("Contact metadata has a different step-pipeline binding")
    contact_bindings = dict(metadata["extractor_code_sha256"])
    contact_bindings["models/contact/deployment_weights.npz"] = metadata["weights_sha256"]
    contact_bindings.update({"models/" + key: value for key, value in metadata["visual_model_sha256"].items()})
    for name, expected in contact_bindings.items():
        if name not in records or records[name]["sha256"] != expected:
            raise ValueError("Contact model/code binding differs: " + name)
    audit = {
        "hotfix_id": HOTFIX_ID,
        "original_metadata_sha256": digest(originals[METADATA]),
        "original_bundle_manifest_sha256": digest(originals[MANIFEST]),
        "source_changes": {name: {"old_sha256": ORIGINAL_HASHES[name],
                                  "new_sha256": PATCHED_HASHES[name],
                                  "changed_function_bodies": ast_changes[name]} for name in replacements},
        "all_other_top_level_ast_and_function_signatures_unchanged": True,
        "contact_extractor_dependencies_other_than_step_pipeline_unchanged": True,
        "contact_weights_and_visual_models_unchanged": True,
        "contact_step_pipeline_helpers_unchanged": True,
        "compatibility_reason": "Contact feature extraction uses unchanged step_pipeline helpers; only fresh scene gating and contact-window admission changed.",
        "geometry_selection_and_abstention_thresholds_unchanged": True,
        "accuracy_improvement_established": False,
    }
    metadata["extractor_code_sha256"]["src/step_pipeline.py"] = PATCHED_HASHES["src/step_pipeline.py"]
    metadata.setdefault("compatibility_updates", []).append(deepcopy(audit))
    replacements[METADATA] = encode_json(metadata)
    updated_manifest = deepcopy(manifest)
    for row in updated_manifest["files"]:
        if row["path"] in replacements:
            data = replacements[row["path"]]
            row.update(bytes=len(data), sha256=digest(data))
    updated_manifest["payload_bytes"] = sum(row["bytes"] for row in updated_manifest["files"])
    updated_manifest.setdefault("compatibility_updates", []).append(deepcopy(audit))
    replacements[MANIFEST] = encode_json(updated_manifest)
    for name, data in replacements.items():
        checksums[name] = digest(data)
    replacements[CHECKSUMS] = ("\n".join(f"{checksums[name]}  {name}" for name in sorted(checksums)) + "\n").encode("utf-8")
    return originals, replacements, audit


def atomic_replace(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".goal-hotfix-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def apply_hotfix(bundle: Path, payload: Path, *, check_only: bool = False) -> dict:
    bundle, payload = bundle.resolve(), payload.resolve()
    originals, replacements, audit = preflight(bundle, payload)
    if check_only:
        return {"status": "ready", "hotfix_id": HOTFIX_ID, "files_to_update": sorted(replacements)}
    for name, original in originals.items():
        if bound_path(bundle, name).read_bytes() != original:
            raise ValueError("Bundle changed during preflight: " + name)
    backup = bundle / BACKUP_NAME
    backup.mkdir()
    for name, data in originals.items():
        destination = backup / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if destination.read_bytes() != data:
            raise OSError("Backup verification failed: " + name)
    (backup / "compatibility_audit.json").write_bytes(encode_json(audit))
    written = []
    try:
        for name, data in replacements.items():
            atomic_replace(bound_path(bundle, name), data)
            written.append(name)
        for name, data in replacements.items():
            if bound_path(bundle, name).read_bytes() != data:
                raise OSError("Post-write verification failed: " + name)
    except BaseException:
        for name in reversed(written):
            atomic_replace(bound_path(bundle, name), originals[name])
        raise
    return {"status": "applied", "hotfix_id": HOTFIX_ID, "backup": str(backup),
            "updated_files": sorted(replacements), "weights_changed": False,
            "accuracy_improvement_established": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-only", action="store_true", help="Validate all files without writing")
    args = parser.parse_args()
    try:
        result = apply_hotfix(args.bundle, args.payload_dir, check_only=args.check_only)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        parser.exit(1, f"Hotfix refused: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
