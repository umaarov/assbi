"""Analytics: line counting, crowd density, anomaly detection, forecasting."""
from .anomaly import AnomalyResult, RollingAnomalyDetector
from .crowd import CrowdAnalyzer, CrowdSnapshot
from .line_counter import LineCounter
from .prediction import Forecast, TrendForecaster, linear_regression

__all__ = [
    "AnomalyResult",
    "RollingAnomalyDetector",
    "CrowdAnalyzer",
    "CrowdSnapshot",
    "LineCounter",
    "Forecast",
    "TrendForecaster",
    "linear_regression",
]
