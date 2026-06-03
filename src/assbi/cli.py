"""Command-line interface for the ASSBI platform.

Examples
--------
Run the dependency-free synthetic demo and populate the warehouse::

    python -m assbi.cli run --session demo

Analyse a real video file with YOLO (requires the optional ML extras)::

    python -m assbi.cli run --source data/source_video.mp4 \
        --backend yolo --render --session street1

Print a report or chat with the assistant about a finished session::

    python -m assbi.cli report --session demo
    python -m assbi.cli chat --session demo

Download the assignment video and launch the BI dashboard::

    python -m assbi.cli download
    python -m assbi.cli dashboard
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from .config import AppConfig, load_env_file
from .persistence.sqlite_repository import SQLiteAnalyticsRepository
from .pipeline.factory import (
    build_assistant,
    build_pipeline,
    build_video_source,
    scale_lines_to_source,
)
from .reporting import ReportBuilder


def _new_session_id(prefix: str = "session") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def cmd_run(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    if args.backend:
        config.detection.backend = args.backend
    if args.render:
        config.video.render = True
    if args.privacy:
        config.privacy.mode = args.privacy
    if getattr(args, "cookies_from_browser", None):
        config.youtube.cookies_from_browser = args.cookies_from_browser
    if getattr(args, "cookies_file", None):
        config.youtube.cookies_file = args.cookies_file
    if args.frames:
        config.video.total_frames = args.frames
        config.video.max_frames = args.frames

    # --stream: pull the assignment video live (no download). Implies YOLO,
    # since the synthetic backend ignores real frames.
    source_arg = args.source
    if args.stream:
        from .video.youtube import DEFAULT_URL

        source_arg = args.source or DEFAULT_URL
        if not args.backend:
            config.detection.backend = "yolo"

    repo = SQLiteAnalyticsRepository(config.database_path)
    try:
        source = build_video_source(config, source_arg)
    except Exception as exc:
        repo.close()
        return _explain_source_error(exc)
    # Scale counting lines to the real frame size (see factory helper).
    scale_lines_to_source(config, source)
    pipeline = build_pipeline(config, repository=repo)
    session_id = args.session or _new_session_id()
    label = source_arg or f"simulation:{config.detection.backend}"
    render_path = config.video.output_path if config.video.render else None

    print(f"▶ Running session '{session_id}' over '{label}' …")
    with source:
        result = pipeline.run(source, session_id, str(label), render_path=render_path)

    s = result.summary
    print("\n✓ Analysis complete")
    print(f"  Frames processed : {s.frames_processed}")
    print(f"  People  IN/OUT   : {s.people_in} / {s.people_out}  (net {s.net_people:+d})")
    print(f"  Vehicles IN/OUT  : {s.vehicles_in} / {s.vehicles_out}")
    print(f"  Peak crowd       : {s.peak_crowd} (frame {s.peak_crowd_frame})")
    print(f"  Anomalies        : {s.anomalies}")
    print(f"  Avg confidence   : {s.avg_confidence:.2f}")
    if render_path:
        print(f"  Annotated video  : {render_path}")
    print(f"\nNext: python -m assbi.cli report --session {session_id}")
    repo.close()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    repo = SQLiteAnalyticsRepository(config.database_path)
    builder = ReportBuilder(repo)
    session_id = args.session or _latest_session(repo)
    if session_id is None:
        print("No sessions found. Run `python -m assbi.cli run` first.")
        return 1

    print(builder.markdown_report(session_id))
    if args.export:
        csv_path = builder.export_frame_csv(session_id, f"data/output/{session_id}_frames.csv")
        json_path = builder.export_summary_json(session_id, f"data/output/{session_id}_summary.json")
        print(f"\nExported:\n  {csv_path}\n  {json_path}")
    repo.close()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    repo = SQLiteAnalyticsRepository(config.database_path)
    session_id = args.session or _latest_session(repo)
    if session_id is None:
        print("No sessions found. Run `python -m assbi.cli run` first.")
        return 1

    assistant = build_assistant(config, repo, session_id)
    mode = "DeepSeek LLM" if assistant.llm is not None else "rule-based (set $DEEPSEEK_API_KEY for AI chat)"
    print(f"🤖 ASSBI assistant ready for session '{session_id}' — {mode}. Type 'help' or 'quit'.\n")
    if args.question:
        print(assistant.ask(args.question).text)
        repo.close()
        return 0
    history: list[tuple[str, str]] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            answer = assistant.ask(q, history=history).text
            history.append(("you", q))
            history.append(("bot", answer))
            print("bot> " + answer + "\n")
    repo.close()
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .video.youtube import DEFAULT_URL, download_video

    url = args.url or DEFAULT_URL
    extra = f" (first {args.duration}s)" if args.duration else ""
    print(f"Downloading {url}{extra} …")
    try:
        path = download_video(
            url,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
            duration=args.duration,
        )
    except Exception as exc:
        return _explain_source_error(exc)
    print(f"✓ Saved to {path}")
    print(f"Now run: python -m assbi.cli run --source {path} --backend yolo --render")
    return 0


def _explain_source_error(exc: Exception) -> int:
    """Turn common (and cryptic) YouTube errors into actionable guidance."""
    msg = str(exc)
    if "not a bot" in msg or "Sign in to confirm" in msg:
        print(
            "\n✗ YouTube blocked the request with a bot check.\n"
            "  Fix: pass your browser cookies, e.g.\n"
            "    --cookies-from-browser edge        (close the browser first!)\n"
            "  or export a cookies.txt and use  --cookies-file cookies.txt"
        )
    elif "cookie database" in msg or "Could not copy" in msg:
        print(
            "\n✗ Couldn't read the browser's cookie database (it's locked while "
            "the browser is open).\n"
            "  Fix A: fully close Edge/Chrome, then re-run with "
            "--cookies-from-browser edge.\n"
            "  Fix B (no need to close it): install a 'Get cookies.txt' browser "
            "extension, export youtube.com cookies, then use "
            "--cookies-file path\\to\\cookies.txt"
        )
    elif ("format is not available" in msg or "JavaScript runtime" in msg
          or "Only images" in msg or "n challenge" in msg):
        print(
            "\n✗ YouTube needs a JavaScript runtime to decode the stream "
            "(the 'n-challenge').\n"
            "  Fix: install Deno —  winget install DenoLand.Deno  — then open a "
            "NEW terminal and re-run. (The app auto-detects Deno once installed.)"
        )
    else:
        print(f"\n✗ Could not open the video source: {msg}")
    return 1


def _add_cookie_args(parser: argparse.ArgumentParser) -> None:
    """Shared --cookies-* flags for commands that hit YouTube."""
    parser.add_argument(
        "--cookies-from-browser", metavar="BROWSER",
        help="Use cookies from this browser (chrome/edge/firefox/brave/…) to "
             "get past YouTube's 'confirm you're not a bot' check",
    )
    parser.add_argument(
        "--cookies-file", metavar="PATH",
        help="Path to a cookies.txt export (alternative to --cookies-from-browser)",
    )


def cmd_dataset(args: argparse.Namespace) -> int:
    import sqlite3

    config = AppConfig.load(args.config)
    conn = sqlite3.connect(config.database_path)
    conn.row_factory = sqlite3.Row
    print(f"=== ASSBI dataset ({config.database_path}) ===\n")
    print("Warehouse tables (structured data derived from video):")
    for table in ("sessions", "frame_analytics", "crossings", "session_summary"):
        n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        print(f"  {table:18}: {n:>7,} rows")

    print("\nPer-session totals (richest first):")
    rows = conn.execute(
        "SELECT session_id, frames_processed, vehicles_in, vehicles_out, "
        "people_in, people_out, peak_crowd FROM session_summary "
        "ORDER BY frames_processed DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r['session_id']:24} {r['frames_processed']:>5}f  "
              f"veh {r['vehicles_in']}/{r['vehicles_out']}  "
              f"ppl {r['people_in']}/{r['people_out']}  peak {r['peak_crowd']}")
    conn.close()
    print("\nExport for Power BI/Excel:  python -m assbi.cli powerbi")
    return 0


def cmd_powerbi(args: argparse.Namespace) -> int:
    from .bi_export import PowerBIExporter

    config = AppConfig.load(args.config)
    out_dir = args.out or "data/output/powerbi"
    print(f"Exporting Power BI star-schema pack from '{config.database_path}' …")
    out, files = PowerBIExporter(config.database_path).export(out_dir)
    print(f"\n✓ Exported to {out}/")
    for f in files:
        print(f"  • {f}")
    print("\nOpen POWERBI_SETUP.md for the build guide (relationships + DAX measures).")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess
    from pathlib import Path

    app = Path(__file__).parent / "dashboard" / "app.py"
    print("Launching Streamlit dashboard … (Ctrl+C to stop)")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def _latest_session(repo: SQLiteAnalyticsRepository) -> str | None:
    sessions = repo.list_sessions()
    return sessions[0].session_id if sessions else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="assbi", description="AI Smart Surveillance BI platform")
    p.add_argument("--config", help="Path to config.yaml / config.json", default="config/config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an analysis session")
    run.add_argument("--source", help="Video path / URL / camera index (omit for synthetic demo)")
    run.add_argument(
        "--stream",
        action="store_true",
        help="Stream the assignment YouTube video live (no download; implies --backend yolo)",
    )
    run.add_argument("--backend", choices=["simulation", "yolo"], help="Detection backend")
    run.add_argument("--session", help="Session id (default: timestamp)")
    run.add_argument("--frames", type=int, help="Limit number of frames")
    run.add_argument("--render", action="store_true", help="Write an annotated output video")
    run.add_argument("--privacy", choices=["off", "blur", "pixelate"],
                     help="Anonymise detected people in the rendered video (GDPR)")
    _add_cookie_args(run)
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="Print / export a session report")
    rep.add_argument("--session", help="Session id (default: latest)")
    rep.add_argument("--export", action="store_true", help="Also export CSV + JSON")
    rep.set_defaults(func=cmd_report)

    chat = sub.add_parser("chat", help="Chat with the analytics assistant")
    chat.add_argument("--session", help="Session id (default: latest)")
    chat.add_argument("--question", help="Ask a single question and exit")
    chat.set_defaults(func=cmd_chat)

    dl = sub.add_parser("download", help="Download the assignment YouTube video (or a clip)")
    dl.add_argument("--url", help="Video URL (default: assignment video)")
    dl.add_argument("--duration", type=float,
                    help="Download only the first N seconds (needs ffmpeg) — recommended for the live cam")
    _add_cookie_args(dl)
    dl.set_defaults(func=cmd_download)

    ds = sub.add_parser("dataset", help="Show a summary of the analytics dataset (warehouse)")
    ds.set_defaults(func=cmd_dataset)

    pbi = sub.add_parser("powerbi", help="Export a Power BI star-schema pack (CSV/Excel + guide)")
    pbi.add_argument("--out", help="Output directory (default: data/output/powerbi)")
    pbi.set_defaults(func=cmd_powerbi)

    dash = sub.add_parser("dashboard", help="Launch the Streamlit BI dashboard")
    dash.set_defaults(func=cmd_dashboard)
    return p


def _force_utf8() -> None:
    """Windows consoles default to cp1252 and crash on Unicode output; make
    stdout/stderr UTF-8 (replacing rather than erroring) for portability."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    load_env_file()  # pick up DEEPSEEK_API_KEY etc. from a .env file if present
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
