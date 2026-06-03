"""Predictive analytics: short-horizon forecasting of footfall / crossings.

Implements ordinary-least-squares linear regression and Holt linear (double
exponential smoothing) in pure Python so the demo forecasts without numpy /
statsmodels. For enterprise use these can be swapped for Prophet or an ARIMA
model behind the same :meth:`forecast` signature.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Forecast:
    method: str
    horizon: int
    predictions: list[float]
    slope: float           # trend per step (people/vehicles per interval)
    r_squared: float       # goodness of fit for the linear model


def linear_regression(y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared) for y against its index."""
    n = len(y)
    if n < 2:
        return 0.0, (y[0] if y else 0.0), 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(y) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (yi - mean_y) for x, yi in zip(xs, y))
    syy = sum((yi - mean_y) ** 2 for yi in y)
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    r_squared = (sxy * sxy) / (sxx * syy) if sxx and syy else 0.0
    return slope, intercept, r_squared


class TrendForecaster:
    """Forecasts the next ``horizon`` values of a 1-D series."""

    def __init__(self, horizon: int = 10) -> None:
        self.horizon = horizon

    def forecast(self, series: list[float], horizon: int | None = None) -> Forecast:
        h = horizon or self.horizon
        if len(series) < 3:
            last = series[-1] if series else 0.0
            return Forecast("naive", h, [last] * h, 0.0, 0.0)

        slope, intercept, r2 = linear_regression(series)
        # Blend the linear trend with Holt smoothing for a stable forecast.
        holt = self._holt(series, h)
        n = len(series)
        preds = []
        for i in range(1, h + 1):
            linear = intercept + slope * (n - 1 + i)
            preds.append(round(max(0.0, 0.5 * linear + 0.5 * holt[i - 1]), 2))
        return Forecast("linear+holt", h, preds, round(slope, 4), round(r2, 3))

    @staticmethod
    def _holt(series: list[float], horizon: int, alpha: float = 0.5, beta: float = 0.3) -> list[float]:
        level = series[0]
        trend = series[1] - series[0]
        for value in series[1:]:
            prev_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        return [level + (i + 1) * trend for i in range(horizon)]
