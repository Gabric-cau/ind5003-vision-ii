"""Verify that a rule replay is bound to an unchanged fresh NN producer."""
from __future__ import annotations

import json
from pathlib import Path

from src.final_results import sha256


def validate_cached_neural_producer(payload, artifact_root):
    if payload.get('new_neural_inference') is True:
        return True
    if payload.get('fresh_neural_cache_used') is not True:
        return False
    if payload.get('new_neural_inference') is not False or payload.get('new_neural_inference_this_run') is not False:
        raise ValueError('Cache replay must explicitly report no new neural inference')
    producer = payload.get('original_neural_inference_producer') or {}
    identity = payload.get('inference_identity') or {}
    original = producer.get('inference_identity') or {}
    for key in ('source_sha256', 'pose_sha256', 'source_manifest_sha256', 'model_sha256'):
        if not identity.get(key) or identity[key] != original.get(key):
            raise ValueError(f'Cache replay producer identity mismatch: {key}')
    if producer.get('new_neural_inference') is not True or producer.get('fresh_model_inference') is not True:
        raise ValueError('Original cache producer must record real fresh inference')
    root = Path(artifact_root).resolve()
    documents = {}
    for label in ('receipt', 'detections'):
        relative = Path(producer.get(f'{label}_file', ''))
        if relative.is_absolute() or relative == Path('.'):
            raise ValueError('Producer artifact paths must be relative')
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Producer artifact is missing or outside its bundle')
        if sha256(path) != producer.get(f'{label}_sha256'):
            raise ValueError(f'Original producer {label} hash mismatch')
        documents[label] = json.loads(path.read_text(encoding='utf-8'))
    receipt, detections = documents['receipt'], documents['detections']
    if any(row.get('clip_id') != payload.get('clip_id') or row.get('inference_identity') != original
           for row in (receipt, detections)):
        raise ValueError('Original producer clip or identity does not match artifacts')
    if receipt.get('new_neural_inference') is not True or receipt.get('reference_labels_used_for_prediction') is not False:
        raise ValueError('Original producer receipt is not fresh source-only inference')
    if detections.get('fresh_model_inference') is not True or detections.get('source_sha256') != identity['source_sha256']:
        raise ValueError('Original detections are not fresh source-bound observations')
    if (receipt.get('source_cache_sha256') != producer['detections_sha256'] or
            payload.get('source_cache_sha256') != producer['detections_sha256']):
        raise ValueError('Cache replay does not reference the original detection bytes')
    return True
