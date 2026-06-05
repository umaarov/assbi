"""Reporting layer: turns the analytics warehouse into KPIs, forecasts and
exportable BI reports (CSV / JSON / Markdown).

This is the "create charts, KPIs, reports" deliverable in code form; the
Streamlit dashboard renders the same numbers visually.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
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


@dataclass
class HourBucket:
    """Crossing counts within one clock hour of the footage timeline."""

    hour: str          # "YYYY-MM-DD HH:00"
    label: str         # human label, e.g. "14:00–15:00"
    people_in: int
    people_out: int

    @property
    def total(self) -> int:
        return self.people_in + self.people_out


@dataclass
class TimeBreakdown:
    """When crossings happened — the temporal view the chatbot reasons over."""

    first: datetime | None
    last: datetime | None
    span_seconds: float
    hours: list[HourBucket] = field(default_factory=list)
    busiest: HourBucket | None = None
    quietest: HourBucket | None = None


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

    # -- temporal analytics ------------------------------------------------
    def time_breakdown(self, session_id: str) -> TimeBreakdown:
        """Bucket crossings by clock hour of the footage timeline.

        Relies on footage-relative timestamps (set by the pipeline from
        frame_index / fps), so the buckets reflect the *video's* hours even
        when the file was processed far faster than real time.
        """
        events = self.repo.crossings(session_id)
        if not events:
            return TimeBreakdown(first=None, last=None, span_seconds=0.0)

        buckets: dict[str, HourBucket] = {}
        first = last = events[0].timestamp
        for ev in events:
            ts = ev.timestamp
            first = min(first, ts)
            last = max(last, ts)
            key = ts.strftime("%Y-%m-%d %H:00")
            b = buckets.get(key)
            if b is None:
                nxt = (ts.hour + 1) % 24
                b = HourBucket(hour=key, label=f"{ts.hour:02d}:00-{nxt:02d}:00",
                               people_in=0, people_out=0)
                buckets[key] = b
            if ev.direction.value == "in":
                b.people_in += 1
            else:
                b.people_out += 1

        hours = [buckets[k] for k in sorted(buckets)]
        busiest = max(hours, key=lambda h: h.total) if hours else None
        quietest = min(hours, key=lambda h: h.total) if hours else None
        return TimeBreakdown(
            first=first, last=last,
            span_seconds=(last - first).total_seconds(),
            hours=hours, busiest=busiest, quietest=quietest,
        )

    def interval_breakdown(self, session_id: str, minutes: float) -> list[dict]:
        """Bucket crossings into fixed ``minutes``-wide bins from the footage
        start. Finer than ``time_breakdown`` (which is hourly) — used to answer
        per-minute questions on short clips."""
        events = self.repo.crossings(session_id)
        if not events:
            return []
        first = min(ev.timestamp for ev in events)
        width = timedelta(minutes=minutes)
        bins: dict[int, list[int]] = {}
        for ev in events:
            idx = int((ev.timestamp - first) / width)
            b = bins.setdefault(idx, [0, 0])
            b[0 if ev.direction.value == "in" else 1] += 1
        rows = []
        for idx in sorted(bins):
            start = first + idx * width
            end = start + width
            rows.append({
                "label": f"{start:%H:%M}-{end:%H:%M}",
                "in": bins[idx][0], "out": bins[idx][1],
                "total": bins[idx][0] + bins[idx][1],
            })
        return rows

    def window_counts(self, session_id: str, minutes: float) -> dict:
        """Crossings within the last ``minutes`` of footage (relative to the
        latest timestamp — for batch files "now" is the end of the video)."""
        events = self.repo.crossings(session_id)
        if not events:
            return {"in": 0, "out": 0, "total": 0, "minutes": minutes, "since": None}
        last = max(ev.timestamp for ev in events)
        cutoff = last - timedelta(minutes=minutes)
        i = sum(1 for ev in events if ev.timestamp >= cutoff and ev.direction.value == "in")
        o = sum(1 for ev in events if ev.timestamp >= cutoff and ev.direction.value == "out")
        return {"in": i, "out": o, "total": i + o, "minutes": minutes, "since": cutoff}

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

    def chatbot_context(self, session_id: str) -> str:
        """The grounding context for the LLM chatbot: the standard report plus a
        temporal breakdown so it can answer time-based questions (which hour was
        busiest, how many crossed in the last N minutes, hourly footfall)."""
        report = self.markdown_report(session_id)
        tb = self.time_breakdown(session_id)
        if not tb.hours:
            return report

        span_min = tb.span_seconds / 60.0
        # Adaptive bucket width: aim for ~30-40 rows so the table fits the
        # context whether the footage is 30 minutes or 12 hours.
        bucket = max(1, round(span_min / 40)) if span_min > 0 else 1
        intervals = self.interval_breakdown(session_id, bucket)
        busiest_in = max(intervals, key=lambda r: r["in"], default=None)
        busiest_out = max(intervals, key=lambda r: r["out"], default=None)

        # "Last N minutes" windows the user commonly asks about.
        windows = [n for n in (5, 10, 15, 30, 60, 120) if n <= span_min + bucket]
        win_lines = []
        for n in windows:
            w = self.window_counts(session_id, n)
            win_lines.append(f"- **Last {n} min of footage:** {w['in']} in / {w['out']} out "
                             f"({w['total']} total)  ")

        lines = [
            report,
            "",
            "## Time breakdown (footage timeline)",
            "",
            f"- **Footage starts:** {tb.first:%Y-%m-%d %H:%M:%S} UTC  ",
            f"- **Footage ends:** {tb.last:%Y-%m-%d %H:%M:%S} UTC  ",
            f"- **Total span:** {span_min:.1f} minutes ({span_min/60:.2f} hours)  ",
            f"- **Busiest hour:** {tb.busiest.label} "
            f"({tb.busiest.total} crossings: {tb.busiest.people_in} in / {tb.busiest.people_out} out)  ",
        ]
        if busiest_in:
            lines.append(f"- **Busiest {bucket}-min period for IN:** {busiest_in['label']} ({busiest_in['in']} in)  ")
        if busiest_out:
            lines.append(f"- **Busiest {bucket}-min period for OUT:** {busiest_out['label']} ({busiest_out['out']} out)  ")
        lines += win_lines
        lines += [
            "",
            f"Footfall in {bucket}-minute periods (use this for 'which minute/period' "
            "or any 'last N minutes' question - sum the relevant rows):",
            "",
            "| Period | People IN | People OUT | Total |",
            "| ------ | --------- | ---------- | ----- |",
        ]
        for r in intervals:
            lines.append(f"| {r['label']} | {r['in']} | {r['out']} | {r['total']} |")
        return "\n".join(lines)
