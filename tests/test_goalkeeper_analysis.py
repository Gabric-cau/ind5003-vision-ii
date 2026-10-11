from src.goalkeeper_analysis import GoalkeeperFeatureExtractor


def trajectory(dx=0.0, dy=0.0, scale=100.0, n=11):
    return [(i / 10, 500 + dx * i / 10, 200 + dy * i / 10, scale) for i in range(n)]


def test_static_goalkeeper_and_sufficient_coverage():
    result = GoalkeeperFeatureExtractor().quantify_pre_kick_movement(trajectory(dx=2, dy=1))
    assert result["gk_prekick_movement_level"] == "static"
    assert result["gk_prekick_coverage"] == 1.0
    assert result["gk_prekick_review_required"] is False


def test_movement_is_scale_invariant():
    small = GoalkeeperFeatureExtractor().quantify_pre_kick_movement(trajectory(dx=70, scale=100))
    large = GoalkeeperFeatureExtractor().quantify_pre_kick_movement(
        [(t, x * 2, y * 2, 200) for t, x, y, _ in trajectory(dx=70, scale=100)]
    )
    assert small["gk_prekick_movement_level"] == "high_active"
    assert large["gk_prekick_movement_level"] == "high_active"
    assert small["gk_prekick_speed_body_s"] == large["gk_prekick_speed_body_s"]


def test_dive_direction_uses_explicit_image_coordinates():
    extractor = GoalkeeperFeatureExtractor()
    assert extractor.classify_dive_direction_and_height(trajectory(dx=-80))["dive_direction"] == "image_left"
    assert extractor.classify_dive_direction_and_height(trajectory(dx=80))["dive_direction"] == "image_right"
    assert extractor.classify_dive_direction_and_height(trajectory(dx=10))["dive_direction"] == "center"


def test_dive_height_respects_screen_y_direction():
    extractor = GoalkeeperFeatureExtractor()
    assert extractor.classify_dive_direction_and_height(trajectory(dx=50, dy=-50))["dive_height"] == "high"
    assert extractor.classify_dive_direction_and_height(trajectory(dx=50, dy=5))["dive_height"] == "mid"
    assert extractor.classify_dive_direction_and_height(trajectory(dx=50, dy=30))["dive_height"] == "low"


def test_insufficient_data_is_unknown_and_reviewable():
    extractor = GoalkeeperFeatureExtractor()
    movement = extractor.quantify_pre_kick_movement([(0.0, 1.0, 2.0, 50.0)])
    dive = extractor.classify_dive_direction_and_height([])
    assert movement["gk_prekick_movement_level"] == "unknown"
    assert dive["dive_direction"] == "unknown"
    assert movement["gk_prekick_review_required"] is True
    assert dive["dive_review_required"] is True
