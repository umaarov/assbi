"""Online anomaly detection on the crowd time-series.

Uses a rolling-window robust z-score (median + MAD) so that a sudden surge or
collapse in the number of people — a stampede, an evacuation, a fight forming a
ring — is flagged in real time without training data. Pure-Python; no numpy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    is_anomaly: bool
    score: float          # robust z-score magnitude
    reason: str


class RollingAnomalyDetector:
    """Flags points whose robust z-score exceeds ``threshold``.

    Args:
        window: number of recent observations forming the baseline.
        threshold: robust z-score above which a point is anomalous (~3.5 is a
            common choice for the MAD-based estimator).
        warmup: minimum observations before any point can be flagged.
    """

    def __init__(
        self,
        window: int = 60,
        threshold: float = 3.5,
        warmup: int = 15,
        min_scale: float = 1.0,
    ) -> None:
        self.window = window
        self.threshold = threshold
        self.warmup = warmup
        # Floor on the estimated spread, in the units of the series (people).
        # Without it, a near-constant low baseline (e.g. 0,0,1,0) gives a MAD of
        # ~0 and any flicker explodes the z-score. A spread below one person is
        # not real signal, so we never divide by less than this.
        self.min_scale = min_scale
        self._buffer: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> AnomalyResult:
        buf = self._buffer
        if len(buf) < self.warmup:
            buf.append(value)
            return AnomalyResult(False, 0.0, "warming-up")

        med = median(buf)
        # Median absolute deviation -> robust estimate of the standard deviation
        # (divide by 0.6745, the inverse normal CDF at 0.75). Floor the estimate
        # at ``min_scale`` so a quiet, near-constant series can't manufacture
        # huge z-scores from one-person fluctuations.
        mad = median([abs(x - med) for x in buf])
        sigma = max(mad / 0.6745, self.min_scale)
        score = (value - med) / sigma
        magnitude = abs(score)
        buf.append(value)

        if magnitude >= self.threshold:
            direction = "surge" if score > 0 else "drop"
            return AnomalyResult(True, magnitude, f"crowd {direction} (z={score:.1f})")
        return AnomalyResult(False, magnitude, "normal")
