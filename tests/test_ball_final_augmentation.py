from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.train_ball_final_augmentation import (
    EpochScheduleSampler, MildContrastBrightness, PositivePreservingTransforms, contained_grid_crops,
    make_training_pool, mixed_schedule, optimizer_budget,
)


def test_grid_crop_preserves_target_exactly_in_source_pixels():
    target = [.53, .52, .01, .02]
    crops = contained_grid_crops(target, 1920, 1080)
    assert crops
    for crop in crops:
        left, top, right, bottom = crop["window"]
        x, y, w, h = crop["normalized"]
        assert [(x * (right - left) + left) / 1920,
                (y * (bottom - top) + top) / 1080,
                w * (right - left) / 1920, h * (bottom - top) / 1080] == pytest.approx(target)
        assert 0 <= x - w / 2 <= x + w / 2 <= 1
        assert 0 <= y - h / 2 <= y + h / 2 <= 1


def test_no_partial_ball_crops_and_no_fake_extra_small_image():
    assert contained_grid_crops([.5, .5, .02, .02], 640, 480) == []
    assert contained_grid_crops([.5, .5, 1., 1.], 1920, 1080) == []
    with pytest.raises(ValueError, match="extends beyond"):
        contained_grid_crops([0., .5, .02, .02], 1920, 1080)


def test_paired_schedule_preserves_source_quota_order_and_budget():
    source = [["old_a", "new_b", "old_c", "new_b"]] * 10
    variants = {"old_a": ["old_a_crop"], "old_c": [], "new_b": ["new_b_crop"]}
    lookup = {name: name for name in variants}
    lookup.update({variant: name for name, values in variants.items() for variant in values})
    mixed = mixed_schedule(source, variants)
    assert [[lookup[name] for name in epoch] for epoch in mixed] == source
    assert mixed == mixed_schedule(source, variants)
    assert any(name.endswith("crop") for epoch in mixed for name in epoch)
    sampler = EpochScheduleSampler(list(lookup), mixed)
    names = list(lookup)
    for epoch in mixed:
        assert [names[i] for i in sampler] == epoch
        assert len(sampler) == 4
    with pytest.raises(RuntimeError, match="unplanned"):
        iter(sampler)


def test_unknown_schedule_entry_or_variable_budget_rejected():
    with pytest.raises(ValueError, match="same budget"):
        EpochScheduleSampler(["a", "b"], [["a"], ["a", "b"]])
    with pytest.raises(ValueError, match="known images"):
        EpochScheduleSampler(["a", "b"], [["c"]])


def test_photometric_transform_cannot_change_target_or_shape():
    image = np.full((30, 40, 3), 100, np.uint8)
    bbox = np.array([[.5, .5, .2, .3]])
    labels = {"img": image, "bboxes": bbox, "im_file": "/tmp/source.jpg"}
    transform = MildContrastBrightness()
    result = transform(labels)
    assert result["bboxes"] is bbox
    assert result["img"].shape == image.shape and result["img"].dtype == np.uint8
    assert not np.array_equal(result["img"], image)
    assert .85 <= transform.observed[0]["contrast"] <= 1.15
    assert -8 <= transform.observed[0]["brightness_255"] <= 8


def test_optimizer_budget_matches_previous_verified_nominal64_protocol():
    assert optimizer_budget(10, 442, nominal_batch=64) == [49, 58, 65, 72, 79, 86, 93, 100, 107, 114]
    assert optimizer_budget(10, 442, nominal_batch=16)[-1] > 114
    assert optimizer_budget(1, 32, nominal_batch=16) == [2]


def test_evaluation_only_source_cannot_enter_export(tmp_path):
    with pytest.raises(ValueError, match="Evaluation-only"):
        make_training_pool([{"split": "test"}], tmp_path / "pool")


def test_infinite_loader_supports_larger_pool_than_epoch_budget():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.data.build import InfiniteDataLoader
    names = ["a", "b", "a_crop", "b_crop"]
    schedules = [["a", "b_crop"], ["a_crop", "b"]]
    loader = InfiniteDataLoader(list(range(4)), batch_size=1, num_workers=0,
        sampler=EpochScheduleSampler(names, schedules), pin_memory=False,
        generator=torch.Generator().manual_seed(29))
    for epoch in schedules:
        assert [names[int(batch[0])] for batch in loader] == epoch
    loader.close()


def test_geometry_drop_restores_positive_without_repeating_photometric():
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import Compose, Format
    from ultralytics.utils.instance import Instances

    def drop_target(labels):
        labels["instances"] = labels["instances"][:0]
        labels["cls"] = labels["cls"][:0]
        return labels

    source = {"img": np.full((40, 80, 3), 100, np.uint8), "im_file": "/tmp/source.jpg",
        "instances": Instances(np.array([[.5, .5, .02, .02]], dtype=np.float32),
                               bbox_format="xywh", normalized=True),
        "cls": np.array([[0]], dtype=np.float32), "resized_shape": (40, 80), "ori_shape": (40, 80)}
    photo = MildContrastBrightness()
    transform = PositivePreservingTransforms(Compose([drop_target, Format(bbox_format="xywh", normalize=True)]), 80, photo)
    result = transform(source)
    assert result["bboxes"].shape == (1, 4)
    assert result["bboxes"].numpy()[0] == pytest.approx([.5, .5, .02, .01], abs=1e-7)
    assert transform.observed == [{"file": "source.jpg", "geometry_fallback": True}]
    assert len(photo.observed) == 1
