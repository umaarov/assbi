"""AI chatbot assistant for the surveillance analytics.

A retrieval-style natural-language interface: the user asks questions in plain
English ("how many cars went out?", "was there an anomaly?") and the assistant
answers from the analytics warehouse. It is intent-matched and deterministic by
default, which makes it reliable and offline. An optional LLM backend can be
plugged in for free-form conversation via the :class:`LLMBackend` protocol —
the rule engine then acts as a grounded tool the LLM can fall back on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..domain.interfaces import AnalyticsRepository
from ..reporting import ReportBuilder


@dataclass
class Answer:
    text: str
    intent: str
    data: dict | None = None


class LLMBackend(Protocol):  # pragma: no cover - integration seam
    def complete(self, system: str, user: str) -> str: ...


class SurveillanceAssistant:
    """Answers questions about a session from the warehouse."""

    def __init__(
        self,
        repository: AnalyticsRepository,
        session_id: str,
        llm: LLMBackend | None = None,
    ) -> None:
        self.repo = repository
        self.session_id = session_id
        self.reports = ReportBuilder(repository)
        self.llm = llm

    def ask(self, question: str, history: list | None = None) -> Answer:
        # When a real LLM is configured, let it drive the conversation — but
        # grounded in this session's analytics so the numbers stay correct.
        if self.llm is not None:
            try:
                return self._llm_answer(question, history)
            except Exception as exc:  # network/key/parse error -> graceful fallback
                rule = self._rule_answer(question)
                rule.data = {**(rule.data or {}), "llm_error": str(exc)}
                return rule
        return self._rule_answer(question)

    def _rule_answer(self, question: str) -> Answer:
        q = question.lower().strip()
        for matcher, handler in self._intents():
            if matcher(q):
                return handler(q)
        return self._fallback(question)

    def _llm_answer(self, question: str, history: list | None) -> Answer:
        context = self.reports.markdown_report(self.session_id)
        system = (
            "You are ASSBI, an AI assistant for a smart-surveillance business-"
            "intelligence platform. You are friendly and concise. Answer the "
            "user's question USING ONLY the analytics report below — never invent "
            "numbers. If they greet you, greet back and say what you can report "
            "on. If the answer isn't in the report, say so briefly.\n\n"
            "=== SESSION ANALYTICS REPORT ===\n" + context
        )
        convo = ""
        for who, msg in (history or [])[-6:]:
            role = "User" if who in ("you", "user") else "Assistant"
            convo += f"{role}: {msg}\n"
        convo += f"User: {question}"
        text = self.llm.complete(system, convo)
        return Answer(text, "llm")

    # -- intent table ------------------------------------------------------
    def _intents(self):
        return [
            (lambda q: _has(q, "help") or _has(q, "what can you"), self._help),
            (lambda q: _people(q) and _has(q, "in", "enter", "entered", "inward"), self._people_in),
            (lambda q: _people(q) and _has(q, "out", "exit", "exited", "left", "outward"), self._people_out),
            (lambda q: _vehicles(q) and _has(q, "in", "enter", "entered", "inward"), self._vehicles_in),
            (lambda q: _vehicles(q) and _has(q, "out", "exit", "exited", "left", "outward"), self._vehicles_out),
            # No direction given -> report both directions combined.
            (lambda q: _people(q), self._people_total),
            (lambda q: _vehicles(q), self._vehicles_total),
            (lambda q: _has(q, "anomaly", "anomalies", "unusual", "abnormal"), self._anomalies),
            (lambda q: _has(q, "peak", "busiest", "most crowded", "maximum"), self._peak),
            (lambda q: _has(q, "forecast", "predict", "next", "future", "trend"), self._forecast),
            (lambda q: _has(q, "breakdown", "by class", "by type", "each"), self._breakdown),
            (lambda q: _has(q, "summary", "overview", "report", "kpi", "total"), self._summary),
            (lambda q: _has(q, "confidence", "accuracy"), self._confidence),
        ]

    # -- handlers ----------------------------------------------------------
    def _summary(self, _q: str) -> Answer:
        kpis = self.reports.kpis(self.session_id)
        if kpis is None:
            return Answer("I don't have any data for this session yet.", "summary")
        text = (
            f"Session overview: {kpis.total_people_in} people entered and "
            f"{kpis.total_people_out} left (net {kpis.net_people:+d}); "
            f"{kpis.total_vehicles_in} vehicles in and {kpis.total_vehicles_out} out. "
            f"Peak crowd was {kpis.peak_crowd}. {kpis.anomaly_count} anomalies were flagged "
            f"across {kpis.frames_processed} frames."
        )
        return Answer(text, "summary", data=vars(kpis))

    def _people_in(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        n = s.people_in if s else 0
        return Answer(f"{n} people crossed inward.", "people_in", {"value": n})

    def _people_out(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        n = s.people_out if s else 0
        return Answer(f"{n} people crossed outward.", "people_out", {"value": n})

    def _vehicles_in(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        n = s.vehicles_in if s else 0
        return Answer(f"{n} vehicles crossed inward.", "vehicles_in", {"value": n})

    def _vehicles_out(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        n = s.vehicles_out if s else 0
        return Answer(f"{n} vehicles crossed outward.", "vehicles_out", {"value": n})

    def _people_total(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        i, o = (s.people_in, s.people_out) if s else (0, 0)
        return Answer(
            f"{i + o} people crossed the line in total ({i} in, {o} out).",
            "people_total", {"in": i, "out": o, "total": i + o},
        )

    def _vehicles_total(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        i, o = (s.vehicles_in, s.vehicles_out) if s else (0, 0)
        return Answer(
            f"{i + o} vehicles crossed the line in total ({i} in, {o} out).",
            "vehicles_total", {"in": i, "out": o, "total": i + o},
        )

    def _anomalies(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        n = s.anomalies if s else 0
        if n == 0:
            return Answer("No anomalies were detected — crowd levels stayed within normal bounds.", "anomalies", {"value": 0})
        frames = [f.frame_index for f in self.repo.frame_series(self.session_id) if f.is_anomaly]
        sample = ", ".join(map(str, frames[:8])) + ("…" if len(frames) > 8 else "")
        return Answer(
            f"{n} anomalous frames were flagged (e.g. frames {sample}). "
            f"These mark sudden surges or drops in crowd density.",
            "anomalies", {"value": n, "frames": frames},
        )

    def _peak(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        if not s:
            return Answer("No data available.", "peak")
        return Answer(
            f"The busiest moment had {s.peak_crowd} people in view (around frame {s.peak_crowd_frame}).",
            "peak", {"peak": s.peak_crowd, "frame": s.peak_crowd_frame},
        )

    def _forecast(self, _q: str) -> Answer:
        f = self.reports.crowd_forecast(self.session_id)
        direction = "rising" if f.slope > 0.01 else "falling" if f.slope < -0.01 else "stable"
        return Answer(
            f"Crowd trend is {direction} ({f.slope:+.3f}/interval, R²={f.r_squared:.2f}). "
            f"Forecast for the next {f.horizon} intervals: {f.predictions[:8]}…",
            "forecast", {"slope": f.slope, "predictions": f.predictions},
        )

    def _breakdown(self, _q: str) -> Answer:
        bd = self.reports.class_breakdown(self.session_id)
        if not bd:
            return Answer("No crossings recorded yet.", "breakdown")
        parts = [f"{cls}: {row['in']} in / {row['out']} out" for cls, row in sorted(bd.items())]
        return Answer("Crossings by class — " + "; ".join(parts) + ".", "breakdown", bd)

    def _confidence(self, _q: str) -> Answer:
        s = self.repo.summary(self.session_id)
        c = s.avg_confidence if s else 0.0
        return Answer(f"Average detection confidence was {c:.0%}.", "confidence", {"value": c})

    def _help(self, _q: str) -> Answer:
        return Answer(
            "Ask me about: people in/out, vehicles in/out, anomalies, the peak crowd, "
            "the crowd forecast, a class breakdown, detection confidence, or a full summary.",
            "help",
        )

    def _fallback(self, _question: str) -> Answer:
        # Reached only in rule-only mode (no LLM configured).
        return Answer(
            "I'm not sure how to answer that. Type 'help' to see what I can report "
            "on — or set $DEEPSEEK_API_KEY to enable free-form AI chat.",
            "unknown",
        )


def _has(text: str, *keywords: str) -> bool:
    # Match the keyword and its simple plural (car -> cars, bus -> buses).
    return any(
        re.search(rf"\b{re.escape(k)}(s|es)?\b", text)
        for k in keywords
    )


def _people(text: str) -> bool:
    return _has(text, "people", "person", "pedestrian", "human", "foot", "footfall")


def _vehicles(text: str) -> bool:
    return _has(text, "car", "vehicle", "truck", "bus", "traffic")
