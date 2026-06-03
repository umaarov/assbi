"""Reporting layer: turns the analytics warehouse into KPIs, forecasts and
exportable BI reports (CSV / JSON / Markdown).

This is the "create charts, KPIs, reports" deliverable in code form; the
Streamlit dashboard renders the same numbers visually.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .analytics.prediction import Forecast, TrendForecaster
from .domain.interfaces import AnalyticsRepository
from .domain.models import SessionSummary


@dataclass
class KPISet:
    """The headline KPIs shown on the dashboard's top row."""

    total_people_in: int
    total_people_out: int
    net_people: int
    total_vehicles_in: int
    total_vehicles_out: int
    total_crossings: int
    peak_crowd: int
    anomaly_count: int
    avg_confidence: float
    frames_processed: int

    @classmethod
    def from_summary(cls, s: SessionSummary) -> "KPISet":
        return cls(
            total_people_in=s.people_in,
            total_people_out=s.people_out,
            net_people=s.net_people,
            total_vehicles_in=s.vehicles_in,
            total_vehicles_out=s.vehicles_out,
            total_crossings=s.total_crossings,
            peak_crowd=s.peak_crowd,
            anomaly_count=s.anomalies,
            avg_confidence=s.avg_confidence,
            frames_processed=s.frames_processed,
        )


class ReportBuilder:
    def __init__(self, repository: AnalyticsRepository) -> None:
        self.repo = repository

    def kpis(self, session_id: str) -> KPISet | None:
        summary = self.repo.summary(session_id)
        return KPISet.from_summary(summary) if summary else None

    def crowd_forecast(self, session_id: str, horizon: int = 25) -> Forecast:
        series = [f.person_count for f in self.repo.frame_series(session_id)]
        return TrendForecaster(horizon).forecast(series)

    def class_breakdown(self, session_id: str) -> dict[str, dict[str, int]]:
        """Counts of crossings per object class, split by direction."""
        out: dict[str, dict[str, int]] = {}
        for event in self.repo.crossings(session_id):
            row = out.setdefault(event.object_class.value, {"in": 0, "out": 0})
            row[event.direction.value] += 1
        return out

    # -- exports -----------------------------------------------------------
    def export_frame_csv(self, session_id: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        series = self.repo.frame_series(session_id)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "frame_index", "timestamp", "person_count", "vehicle_count",
                "total_detections", "crossings_in", "crossings_out",
                "is_anomaly", "anomaly_score",
            ])
            for f in series:
                writer.writerow([
                    f.frame_index, f.timestamp.isoformat(), f.person_count,
                    f.vehicle_count, f.total_detections, f.crossings_in,
                    f.crossings_out, int(f.is_anomaly), f.anomaly_score,
                ])
        return path

    def export_summary_json(self, session_id: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.repo.summary(session_id)
        kpis = self.kpis(session_id)
        forecast = self.crowd_forecast(session_id)
        payload = {
            "session": asdict(summary) if summary else None,
            "kpis": asdict(kpis) if kpis else None,
            "class_breakdown": self.class_breakdown(session_id),
            "crowd_forecast": asdict(forecast),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def markdown_report(self, session_id: str) -> str:
        summary = self.repo.summary(session_id)
        if summary is None:
            return f"No data for session `{session_id}`."
        kpis = KPISet.from_summary(summary)
        forecast = self.crowd_forecast(session_id)
        breakdown = self.class_breakdown(session_id)

        lines = [
            f"# ASSBI Analytics Report — `{session_id}`",
            "",
            f"**Source:** {summary.source}  ",
            f"**Frames processed:** {summary.frames_processed}  ",
            f"**Video duration:** {summary.duration_seconds:.1f}s  ",
            "",
            "## Key Performance Indicators",
            "",
            "| KPI | Value |",
            "| --- | ----- |",
            f"| People IN | {kpis.total_people_in} |",
            f"| People OUT | {kpis.total_people_out} |",
            f"| Net people | {kpis.net_people} |",
            f"| Vehicles IN | {kpis.total_vehicles_in} |",
            f"| Vehicles OUT | {kpis.total_vehicles_out} |",
            f"| Total crossings | {kpis.total_crossings} |",
            f"| Peak crowd (frame {summary.peak_crowd_frame}) | {kpis.peak_crowd} |",
            f"| Anomalies flagged | {kpis.anomaly_count} |",
            f"| Avg detection confidence | {kpis.avg_confidence:.2f} |",
            "",
            "## Crossings by object class",
            "",
            "| Class | IN | OUT |",
            "| ----- | -- | --- |",
        ]
        for cls, row in sorted(breakdown.items()):
            lines.append(f"| {cls} | {row['in']} | {row['out']} |")
        lines += [
            "",
            "## Predictive analytics",
            "",
            f"- **Model:** {forecast.method}",
            f"- **Trend (people / interval):** {forecast.slope:+.3f} "
            f"(R² = {forecast.r_squared:.2f})",
            f"- **Next {forecast.horizon}-interval crowd forecast:** "
            f"{forecast.predictions}",
        ]
        return "\n".join(lines)
