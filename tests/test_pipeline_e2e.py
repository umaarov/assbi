"""End-to-end test of the synthetic pipeline through to persistence + chatbot."""
import tempfile
from pathlib import Path

from assbi.chatbot.assistant import SurveillanceAssistant
from assbi.config import AppConfig
from assbi.persistence.sqlite_repository import SQLiteAnalyticsRepository
from assbi.pipeline.factory import build_pipeline, build_video_source
from assbi.reporting import ReportBuilder


def _run(tmpdir: str):
    config = AppConfig()
    config.database_path = str(Path(tmpdir) / "test.db")
    config.video.total_frames = 200
    repo = SQLiteAnalyticsRepository(config.database_path)
    pipeline = build_pipeline(config, repository=repo)
    source = build_video_source(config, None)
    with source:
        result = pipeline.run(source, "t1", "simulation:test")
    return repo, result


def test_pipeline_populates_warehouse():
    with tempfile.TemporaryDirectory() as tmp:
        repo, result = _run(tmp)
        s = result.summary
        assert s.frames_processed == 200
        # synthetic scene is busy: there must be crossings and detections
        assert s.total_crossings > 0
        assert len(repo.frame_series("t1")) == 200
        assert repo.summary("t1") is not None
        repo.close()


def test_report_and_chatbot():
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _run(tmp)
        builder = ReportBuilder(repo)
        md = builder.markdown_report("t1")
        assert "Key Performance Indicators" in md

        bot = SurveillanceAssistant(repo, "t1")
        assert "people" in bot.ask("how many people went in?").text.lower()
        assert bot.ask("summary").intent == "summary"
        assert bot.ask("forecast").intent == "forecast"
        # Direction-less questions report the combined total, not a fallback.
        cars = bot.ask("how many cars crossed?")
        assert cars.intent == "vehicles_total"
        assert cars.data["total"] == cars.data["in"] + cars.data["out"]
        assert bot.ask("how many people crossed the line?").intent == "people_total"
        # A direction word still wins over the total handler.
        assert bot.ask("how many cars went out?").intent == "vehicles_out"
        repo.close()


class _FakeLLM:
    """Records the grounded prompt and returns a canned reply."""
    def __init__(self) -> None:
        self.system = None
        self.user = None

    def complete(self, system: str, user: str) -> str:
        self.system = system
        self.user = user
        return "Hi! I can report on people, vehicles, anomalies and forecasts."


def test_llm_assistant_is_grounded_and_handles_freeform():
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _run(tmp)
        fake = _FakeLLM()
        bot = SurveillanceAssistant(repo, "t1", llm=fake)

        # Free-form input the rule engine would have rejected now routes to the LLM.
        ans = bot.ask("Hello", history=[])
        assert ans.intent == "llm"
        assert "report on" in ans.text.lower()
        # The model was grounded with the real session report (KPIs present).
        assert "Key Performance Indicators" in fake.system
        # History is threaded into the conversation.
        bot.ask("tell me abt sumary", history=[("you", "Hello"), ("bot", ans.text)])
        assert "Assistant:" in fake.user and "User: tell me abt sumary" in fake.user
        repo.close()


def test_llm_failure_falls_back_to_rules():
    class _BrokenLLM:
        def complete(self, system, user):
            raise RuntimeError("network down")

    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _run(tmp)
        bot = SurveillanceAssistant(repo, "t1", llm=_BrokenLLM())
        ans = bot.ask("how many cars crossed?")
        # Falls back to the deterministic engine rather than erroring out.
        assert ans.intent == "vehicles_total"
        repo.close()


def test_rerun_session_is_idempotent():
    # Re-running the same session id must replace its facts, not append them,
    # so the per-class crossing breakdown can't double-count.
    with tempfile.TemporaryDirectory() as tmp:
        config = AppConfig()
        config.database_path = str(Path(tmp) / "test.db")
        config.video.total_frames = 200
        repo = SQLiteAnalyticsRepository(config.database_path)

        for _ in range(2):
            pipeline = build_pipeline(config, repository=repo)
            source = build_video_source(config, None)
            with source:
                result = pipeline.run(source, "dup", "simulation:test")

        # Crossings stored for the session equal exactly one run's worth.
        stored = len(repo.crossings("dup"))
        assert stored == result.summary.total_crossings
        assert len(repo.frame_series("dup")) == 200
        repo.close()


def test_crossings_have_both_classes():
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _run(tmp)
        bd = ReportBuilder(repo).class_breakdown("t1")
        # The synthetic scene generates both pedestrians and vehicles.
        assert "person" in bd
        assert any(k in bd for k in ("car", "truck", "bus"))
        repo.close()
