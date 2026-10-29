import pytest

from scripts.train_external_ball_round1 import TrainingEvidence, partition_samples


def samples():
    old = [{"video_id": v, "split": "train" if v != "match_01" else "validation", "image_sha256": v}
           for v in ("core_01", "core_02", "core_03", "match_01")]
    new = [{"video_id": m + "_p01_s0003", "match_id": m, "event_id": m + "_p01",
            "split": s, "image_sha256": m}
           for m, s in [("lev_stu_20240817", "train"), ("ars_mci_20230806", "validation")]]
    regression = [{"video_id": v, "split": "test", "image_sha256": v}
                  for v in ("core_05", "core_06", "core_09")]
    return old, new, regression


def test_complete_new_match_is_excluded_from_training():
    old, new, regression = samples()
    train, held = partition_samples(old, new, regression)
    assert len(train) == 5
    assert held == [new[1]]
    assert not any(r["video_id"].startswith("ars_mci") for r in train)


def test_replayed_view_cannot_cross_match_split():
    old, new, regression = samples()
    new.append({**new[0], "video_id": "lev_stu_20240817_p01_s0009", "split": "validation"})
    with pytest.raises(ValueError, match="entire new match"):
        partition_samples(old, new, regression)


def test_renaming_identical_image_does_not_bypass_split_check():
    old, new, regression = samples()
    new[0]["image_sha256"] = regression[0]["image_sha256"]
    with pytest.raises(ValueError, match="Identical source images"):
        partition_samples(old, new, regression)


def test_old_evaluation_event_cannot_be_added_to_training():
    old, new, regression = samples()
    old.append(regression[0])
    with pytest.raises(ValueError, match="Unexpected original development"):
        partition_samples(old, new, regression)


def test_optimizer_updates_and_recovery_copy_are_verified(tmp_path):
    import json
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    trainer = SimpleNamespace(model=model, optimizer=optimizer, device=torch.device("cpu"),
                              batch_size=1, epoch=0, epochs=30, _oom_retries=0,
                              last=tmp_path / "last.pt")
    evidence = TrainingEvidence(tmp_path, 1, "cpu")
    evidence.start(trainer)
    loss = model(torch.ones(1, 2)).square().mean() + model.weight.square().mean()
    trainer.loss_items = trainer.tloss = {"loss": loss.detach()}
    loss.backward()
    optimizer.step()
    evidence.batch_finished(trainer)
    torch.save({"epoch": 0, "optimizer": optimizer.state_dict()}, trainer.last)
    evidence.saved(trainer)
    state = json.loads((tmp_path / "training_progress.json").read_text())
    assert state["optimizer_steps"] == 1
    assert state["maximum_observed_parameter_delta"] > 0
    assert state["completed_epochs"] == [1]
    recovered = tmp_path / "last_recoverable.pt"
    assert recovered.read_bytes() == trainer.last.read_bytes()
    torch.save({"epoch": 1}, trainer.last)
    assert torch.load(recovered, weights_only=True)["epoch"] == 0
    evidence.handle.remove()


def test_changed_batch_and_nonfinite_loss_stop_training(tmp_path):
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    evidence = TrainingEvidence(tmp_path, 1, "cpu")
    trainer = SimpleNamespace(batch_size=2, _oom_retries=0, loss_items={"loss": torch.tensor(1.)})
    with pytest.raises(RuntimeError, match="batch changed"):
        evidence.batch_finished(trainer)
    trainer.batch_size = 1
    trainer.loss_items["loss"] = torch.tensor(float("nan"))
    with pytest.raises(RuntimeError, match="Non-finite training loss"):
        evidence.batch_finished(trainer)
