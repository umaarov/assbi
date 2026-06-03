from assbi.analytics.anomaly import RollingAnomalyDetector
from assbi.analytics.prediction import TrendForecaster, linear_regression


def test_anomaly_warmup_then_flags_surge():
    det = RollingAnomalyDetector(window=30, threshold=3.5, warmup=10)
    # steady baseline
    for _ in range(20):
        res = det.update(5.0)
    assert res.is_anomaly is False
    # sudden surge
    spike = det.update(60.0)
    assert spike.is_anomaly is True
    assert spike.score >= 3.5


def test_anomaly_ignores_normal_variation():
    det = RollingAnomalyDetector(window=30, threshold=3.5, warmup=10)
    flagged = 0
    for i in range(60):
        value = 10 + (i % 3)  # small oscillation
        if det.update(float(value)).is_anomaly:
            flagged += 1
    assert flagged == 0


def test_anomaly_quiet_low_count_does_not_overfire():
    # Near-constant low counts (a calm street: 0,0,1,0,1...) collapse the MAD to
    # ~0. Without a spread floor this manufactures huge z-scores; with it, these
    # one-person flickers must stay below threshold.
    det = RollingAnomalyDetector(window=90, threshold=4.0, warmup=30, min_scale=1.0)
    flagged = 0
    pattern = [0, 0, 1, 0, 1, 0, 2, 1, 0, 0]
    for i in range(300):
        if det.update(float(pattern[i % len(pattern)])).is_anomaly:
            flagged += 1
    assert flagged == 0
    # A genuine surge on that quiet baseline is still caught.
    assert det.update(15.0).is_anomaly is True


def test_linear_regression_perfect_fit():
    slope, intercept, r2 = linear_regression([0, 2, 4, 6, 8])
    assert abs(slope - 2.0) < 1e-9
    assert abs(intercept - 0.0) < 1e-9
    assert abs(r2 - 1.0) < 1e-9


def test_forecast_extends_trend():
    f = TrendForecaster(horizon=3).forecast([1, 2, 3, 4, 5])
    assert len(f.predictions) == 3
    # rising series -> next values should exceed the last observed
    assert f.predictions[0] >= 5
    assert f.slope > 0


def test_forecast_handles_short_series():
    f = TrendForecaster(horizon=4).forecast([7])
    assert f.predictions == [7, 7, 7, 7]
