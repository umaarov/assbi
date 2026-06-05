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
    if getattr(args, "stride", None):
        config.video.stride = args.stride

    start_time = None
    if getattr(args, "start_time", None):
        from datetime import datetime, timezone

        start_time = datetime.fromisoformat(args.start_time)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

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
        result = pipeline.run(source, session_id, str(label), render_path=render_path,
                              start_time=start_time)

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
            max_height=args.max_height,
        )
    except Exception as exc:
        return _explain_source_error(exc)
    print(f"✓ Saved to {path}")
    print(f"Now run: python -m assbi.cli run --source {path} --backend yolo --render")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Record N seconds of the LIVE stream to a file in real time, via ffmpeg.

    Unlike ``download`` (which can only grab the cam's short rewind buffer), this
    follows the live edge and captures as the stream airs, so it can produce
    hours of footage — it just takes that long in wall-clock time.
    """
    import os
    import subprocess

    from .video.youtube import DEFAULT_URL, ensure_js_runtime, find_ffmpeg, stream_url

    config = AppConfig.load(args.config)
    url = args.url or DEFAULT_URL
    ensure_js_runtime()
    ff_dir = find_ffmpeg()
    if not ff_dir:
        print("✗ ffmpeg not found. Install it (winget install Gyan.FFmpeg) or place "
              "ffmpeg.exe in ./tools/.")
        return 1
    ffmpeg = os.path.join(ff_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    try:
        media = stream_url(
            url,
            max_height=args.max_height,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
            remote_components=config.youtube.remote_components,
        )
    except Exception as exc:
        return _explain_source_error(exc)

    out = args.out or "data/source_video.mp4"
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mins = args.duration / 60.0
    print(f"● Recording the live stream to {out} for {args.duration:.0f}s (~{mins:.0f} min) …")
    print("  This runs in real time — leave it until it finishes (Ctrl+C stops early but keeps what's recorded).")
    cmd = [
        ffmpeg, "-y",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", media,
        "-t", str(args.duration),
        "-an", "-c:v", "copy",          # drop audio, copy video (no re-encode)
        "-loglevel", "warning", "-stats",
        out,
    ]
    rc = subprocess.call(cmd)
    if rc == 0 or os.path.isfile(out):
        print(f"\n✓ Saved {out}")
        print(f"Now process it:  python -m assbi.cli run --source {out} "
              f"--backend yolo --session dublin_2h --stride 8")
        return 0
    print(f"✗ ffmpeg exited with {rc}")
    return rc


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Sample frames from footage and auto-label them into a YOLO dataset."""
    from .training import build_dataset

    try:
        stats = build_dataset(
            source=args.source,
            out_dir=args.out,
            n_frames=args.frames,
            val_split=args.val_split,
            label_model=args.label_model,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"✗ {exc}")
        return 1
    print(f"\nReview/correct labels if needed, then train:\n"
          f"  python -m assbi.cli train --data {stats.data_yaml}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Fine-tune YOLO on the built dataset."""
    from .training import train_model

    try:
        result = train_model(
            data_yaml=args.data,
            base=args.base,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            device=args.device,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"✗ {exc}")
        return 1
    print("\n✓ Training complete")
    print(f"  Best weights : {result.best_weights}")
    print(f"  Run dir      : {result.run_dir}  (curves, confusion matrix, results.csv)")
    if result.metrics:
        for k, v in result.metrics.items():
            print(f"  {k:14}: {v}")
    print("\nTo use your model, set in config.yaml:")
    print(f"  detection.model_path: {result.best_weights.as_posix()}")
    return 0


def cmd_train_chatbot(args: argparse.Namespace) -> int:
    """Train the intent classifier for the chatbot's NLU."""
    if args.backend == "bow":
        from .chatbot.intent_model import train

        print("Training NLU (bag-of-words → MLP) …")
        try:
            r = train(epochs=args.epochs, hidden=args.hidden, lr=args.lr)
        except RuntimeError as exc:
            print(f"✗ {exc}")
            return 1
        print("\n✓ Chatbot NLU trained (bag-of-words)")
        print(f"  Test accuracy : {r.accuracy:.1%}")
        print(f"  Macro F1      : {r.macro_f1:.3f}")
        print(f"  Train / test  : {r.n_train} / {r.n_test} · {r.n_intents} intents")
        print(f"  Model         : {r.model_path}")
    else:
        from .chatbot.intent_nlu import train

        print("Training NLU via transfer learning (MiniLM embeddings → neural head) …")
        try:
            r = train(epochs=args.epochs, hidden=args.hidden, lr=args.lr)
        except Exception as exc:
            print(f"✗ {exc}\n  (Needs sentence-transformers: pip install sentence-transformers, "
                  "or use --backend bow.)")
            return 1
        print("\n✓ Chatbot NLU trained (transfer learning)")
        print(f"  Test accuracy        : {r.test_accuracy:.1%}")
        print(f"  Macro F1             : {r.macro_f1:.3f}")
        print(f"  5-fold CV accuracy   : {r.cv_mean:.1%} ± {r.cv_std:.1%}")
        print(f"  Hard novel-paraphrase: {r.hard_accuracy:.1%}  (unseen wordings)")
        print(f"  Train / test         : {r.n_train} / {r.n_test} · {r.n_intents} intents")
        print(f"  Head weights         : {r.head_path}")
        print(f"  Dataset              : {r.dataset_path}")
        print(f"  Metrics              : {r.metrics_path}")
        if r.confusion_path:
            print(f"  Confusion plot       : {r.confusion_path}")
    print("\nThe assistant auto-loads it on next run (cli chat / dashboard).")
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
    run.add_argument("--stride", type=int,
                     help="Process every Nth frame (higher = cover more footage faster; e.g. 10 for a long capture)")
    run.add_argument("--start-time", metavar="ISO",
                     help="Anchor footage timestamps, e.g. 2026-06-04T08:00 (default: now). "
                          "Frames are stamped start-time + frame_index/fps, so the warehouse "
                          "carries the video's real timeline for hourly/time queries.")
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
    dl.add_argument("--max-height", type=int, default=360,
                    help="Cap download resolution (default 360 keeps a 30-60 min clip small & fast to process)")
    _add_cookie_args(dl)
    dl.set_defaults(func=cmd_download)

    rec = sub.add_parser("record", help="Record N seconds of the live stream to a file in real time (needs ffmpeg)")
    rec.add_argument("--url", help="Video URL (default: assignment video)")
    rec.add_argument("--duration", type=float, default=7200.0,
                     help="Seconds to record in real time (default 7200 = 2 hours)")
    rec.add_argument("--max-height", type=int, default=360,
                     help="Cap recording resolution (default 360)")
    rec.add_argument("--out", help="Output file (default data/source_video.mp4)")
    _add_cookie_args(rec)
    rec.set_defaults(func=cmd_record)

    ds = sub.add_parser("dataset", help="Show a summary of the analytics dataset (warehouse)")
    ds.set_defaults(func=cmd_dataset)

    bd = sub.add_parser("build-dataset",
                        help="Sample frames from footage and auto-label them into a YOLO training dataset")
    bd.add_argument("--source", default="data/source_video.mp4", help="Video file to sample")
    bd.add_argument("--frames", type=int, default=500, help="Number of labelled frames to keep")
    bd.add_argument("--out", default="data/dataset", help="Dataset output directory")
    bd.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (default 0.2)")
    bd.add_argument("--label-model", default="yolov8s.pt",
                    help="Pretrained 'teacher' weights for auto-labelling (default yolov8s.pt)")
    bd.add_argument("--conf", type=float, default=0.30, help="Min confidence to keep a label")
    bd.add_argument("--imgsz", type=int, default=640, help="Teacher inference size (default 640)")
    bd.set_defaults(func=cmd_build_dataset)

    tc = sub.add_parser("train-chatbot",
                        help="Train the neural intent classifier for the chatbot's NLU")
    tc.add_argument("--backend", choices=["transfer", "bow"], default="transfer",
                    help="transfer = MiniLM embeddings + neural head (default, best); "
                         "bow = lightweight bag-of-words (no extra deps)")
    tc.add_argument("--epochs", type=int, default=300, help="Training epochs (default 300)")
    tc.add_argument("--hidden", type=int, default=128, help="Hidden layer size (default 128)")
    tc.add_argument("--lr", type=float, default=0.01, help="Learning rate (default 0.01)")
    tc.set_defaults(func=cmd_train_chatbot)

    tr = sub.add_parser("train", help="Fine-tune YOLO on the built dataset")
    tr.add_argument("--data", default="data/dataset/data.yaml", help="Path to data.yaml")
    tr.add_argument("--base", default="yolov8n.pt", help="Base weights to fine-tune (default yolov8n.pt)")
    tr.add_argument("--epochs", type=int, default=30, help="Training epochs (default 30)")
    tr.add_argument("--imgsz", type=int, default=416, help="Training image size (default 416)")
    tr.add_argument("--batch", type=int, default=8, help="Batch size (default 8)")
    tr.add_argument("--name", default="finetune", help="Run name under runs/assbi/")
    tr.add_argument("--device", default=None, help="cpu (default) or GPU index like 0")
    tr.set_defaults(func=cmd_train)

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
