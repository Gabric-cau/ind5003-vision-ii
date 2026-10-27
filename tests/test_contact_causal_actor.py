from src.contact_causal_actor import recent_stationary_support, predict_causal_contact


def test_future_observation_cannot_supply_precontact_evidence():
    candidate = {"contact_frame": 100, "stationary_ball_observations": [{"frame": 50}, {"frame": 102}]}
    result = recent_stationary_support(candidate, 50.)
    assert result["gap_seconds"] == 1.
    assert result["supported"] is False


def test_recent_precontact_evidence_is_supported():
    candidate = {"contact_frame": 100, "stationary_ball_observations": [{"frame": 90}, {"frame": 102}]}
    assert recent_stationary_support(candidate, 50.)["supported"] is True


def test_failed_causal_gate_abstains_without_invoking_model():
    record = {"diagnostics": {"causal_actor_recovery": {"status": "abstain"}}}
    assert predict_causal_contact(record, {})["time_seconds"] is None
