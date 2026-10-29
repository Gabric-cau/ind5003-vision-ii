from collections import Counter

import pytest

from scripts.train_ball_controlled_round2 import ScheduledSampler, make_schedules, sequence_digest


def data():
    old = [f"old_{i:03}.jpg" for i in range(122)]
    new = [f"new_{i:03}.jpg" for i in range(320)]
    return old + new, set(old)


def test_natural_keeps_all_images_once_per_epoch():
    names, old = data()
    schedule = make_schedules(names, old, balanced=False)
    assert len(schedule) == 10
    assert all(Counter(epoch) == Counter(names) for epoch in schedule)


def test_balanced_budget_is_exact_and_covers_new_images_over_round():
    names, old = data()
    schedule = make_schedules(names, old, balanced=True)
    assert all(len(epoch) == 442 and sum(name in old for name in epoch) == 221 for epoch in schedule)
    assert set(sum(schedule, [])) == set(names)
    assert schedule == make_schedules(list(reversed(names)), old, balanced=True)
    assert sequence_digest(schedule) == sequence_digest(make_schedules(names, old, balanced=True))


def test_sampler_binds_names_after_framework_sorting_and_stops():
    names, old = data()
    schedule = make_schedules(names, old, balanced=True)
    loaded = list(reversed(names))
    sampler = ScheduledSampler(loaded, schedule)
    for epoch in schedule:
        assert [loaded[index] for index in sampler] == epoch
        assert len(sampler) == 442
    with pytest.raises(RuntimeError, match="unplanned epoch"):
        iter(sampler)


def test_bad_identity_or_quota_cannot_silently_change_experiment():
    with pytest.raises(ValueError, match="Unique names"):
        make_schedules(["a", "a"], {"a"}, balanced=True)
    with pytest.raises(ValueError, match="even sample"):
        make_schedules(["a", "b", "c"], {"a"}, balanced=True)
    with pytest.raises(ValueError, match="unknown images"):
        ScheduledSampler(["a", "b"], [["a", "missing"]])


def test_infinite_loader_consumes_exactly_one_frozen_schedule_per_epoch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.data.build import InfiniteDataLoader
    names = ["a", "b", "c", "d"]
    schedule = make_schedules(names, {"a"}, balanced=True, epochs=3)
    sampler = ScheduledSampler(names, schedule)
    loader = InfiniteDataLoader(list(range(4)), batch_size=1, num_workers=0,
                                sampler=sampler, pin_memory=False,
                                generator=torch.Generator().manual_seed(27))
    for epoch in schedule:
        assert [names[int(batch[0])] for batch in loader] == epoch
    assert sampler.next_epoch == 3
    loader.close()
