"""End-to-end synthetic demo: runs the pipeline, prints a report, and shows the
chatbot answering questions — all with zero third-party dependencies.

Usage:
    python scripts/run_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assbi.chatbot.assistant import SurveillanceAssistant  # noqa: E402
from assbi.config import AppConfig  # noqa: E402
from assbi.persistence.sqlite_repository import SQLiteAnalyticsRepository  # noqa: E402
from assbi.pipeline.factory import build_pipeline, build_video_source  # noqa: E402
from assbi.reporting import ReportBuilder  # noqa: E402


def main() -> None:
    config = AppConfig()                      # defaults — pure synthetic demo
    config.database_path = "data/assbi.db"
    config.video.total_frames = 600

    repo = SQLiteAnalyticsRepository(config.database_path)
    pipeline = build_pipeline(config, repository=repo)
    source = build_video_source(config, None)  # None => synthetic source
    session_id = "demo"

    print("Running synthetic surveillance pipeline (600 frames)…\n")
    with source:
        pipeline.run(source, session_id, "simulation:street-demo")

    builder = ReportBuilder(repo)
    print(builder.markdown_report(session_id))

    print("\n--- AI assistant Q&A ---")
    assistant = SurveillanceAssistant(repo, session_id)
    for q in [
        "Give me a summary",
        "How many people went in?",
        "How many cars crossed out?",
        "Were there any anomalies?",
        "What's the crowd forecast?",
    ]:
        print(f"\nQ: {q}\nA: {assistant.ask(q).text}")

    builder.export_summary_json(session_id, "data/output/demo_summary.json")
    builder.export_frame_csv(session_id, "data/output/demo_frames.csv")
    print("\nExported data/output/demo_summary.json and demo_frames.csv")
    repo.close()


if __name__ == "__main__":
    main()
