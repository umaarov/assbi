"""Streamlit Business-Intelligence dashboard for ASSBI.

Run with::

    streamlit run src/assbi/dashboard/app.py
    # or:  python -m assbi.cli dashboard

A single command-center over the analytics warehouse:

* **Live Monitor** — runs YOLO on the live stream and renders the annotated
  video in real time with live IN/OUT counters (reuses the pipeline directly).
* **Overview / Trends / Crossings / Forecast / Anomalies** — BI views of any
  recorded session: KPI cards, time-series, class breakdowns, the predictive
  forecast and flagged anomalies.
* **Assistant** — the embedded natural-language analytics chatbot.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run src/assbi/dashboard/app.py` without installing the pkg.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from assbi.config import AppConfig, load_env_file  # noqa: E402
from assbi.persistence.sqlite_repository import SQLiteAnalyticsRepository  # noqa: E402
from assbi.reporting import KPISet, ReportBuilder  # noqa: E402

load_env_file()  # pick up DEEPSEEK_API_KEY from a .env file if present
st.set_page_config(page_title="ASSBI — Smart Surveillance BI", layout="wide", page_icon="🛰️")

ASSIGNMENT_URL = "https://www.youtube.com/watch?v=3nyPER2kzqk"  # EarthCam Live: Dublin, Ireland


@st.cache_resource
def get_repo(db_path: str) -> SQLiteAnalyticsRepository:
    return SQLiteAnalyticsRepository(db_path)


def main() -> None:
    st.title("🛰️ ASSBI — AI Smart Surveillance Business Intelligence")
    st.caption("Real-time human & vehicle analytics · YOLO + OpenCV · line-crossing counts · predictive BI")

    config = AppConfig.load("config/config.yaml")
    repo = get_repo(config.database_path)
    sessions = repo.list_sessions()
    builder = ReportBuilder(repo)

    with st.sidebar:
        st.header("📁 Sessions")
        # Default to the richest session (most frames) so the BI tabs open on
        # real data rather than a small test run.
        ordered = sorted(sessions, key=lambda s: s.frames_processed, reverse=True)
        ids = [s.session_id for s in ordered]
        session_id = st.selectbox("Active session (for BI views)", ids) if ids else None
        st.divider()
        st.caption(f"Warehouse: `{config.database_path}`")
        for s in ordered:
            st.text(f"• {s.session_id} — {s.frames_processed} frames")
        if not sessions:
            st.info("No recorded sessions yet. Use the Live Monitor tab to create one.")

    tabs = st.tabs([
        "🔴 Live Monitor", "📊 Overview", "📋 Dataset", "📈 Trends",
        "🚦 Crossings", "🔮 Forecast", "⚠️ Anomalies", "🤖 Assistant",
    ])

    with tabs[0]:
        _render_live(config, repo)

    if session_id is None:
        for t in tabs[1:]:
            with t:
                st.info("Run a session in the Live Monitor tab first.")
        return

    summary = repo.summary(session_id)
    kpis = KPISet.from_summary(summary)
    series = repo.frame_series(session_id)

    with tabs[1]:
        st.subheader(f"Session `{session_id}` — {summary.source}")
        _render_kpis(kpis)
        _render_export(config)
        _render_recording()
    with tabs[2]:
        _render_dataset(config, session_id)
    with tabs[3]:
        _render_timeseries(series)
    with tabs[4]:
        _render_breakdown(builder, session_id)
    with tabs[5]:
        _render_forecast(builder, session_id)
    with tabs[6]:
        _render_anomalies(series)
    with tabs[7]:
        _render_assistant(config, repo, session_id)


# --------------------------------------------------------------------------- #
#  Live monitor                                                               #
# --------------------------------------------------------------------------- #
def _render_live(config: AppConfig, repo: SQLiteAnalyticsRepository) -> None:
    st.subheader("🔴 Live YOLO monitor")
    st.caption(
        "Streams the source through the real pipeline (YOLO → tracking → "
        "line-counting) and renders the annotated video live. Results are saved "
        "to the warehouse and appear in the other tabs."
    )

    try:
        import cv2  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        st.error(
            "The live monitor needs the CV stack. Install it with:\n\n"
            "```\npip install -r requirements-full.txt\n```\n\n"
            "(This is a local-only feature — the deployed cloud app serves the "
            "recorded warehouse instead.)"
        )
        return

    local_clip = "data/source_video.mp4"
    have_clip = Path(local_clip).exists()

    c1, c2, c3 = st.columns([2, 1, 1])
    source_choice = c1.selectbox(
        "Source",
        [
            "Local clip — data/source_video.mp4 (fast)",
            "Assignment stream — Dublin, Ireland (live, slower)",
            "Custom URL / RTSP / file path",
            "Webcam 0",
        ],
    )
    custom_url = ""
    if source_choice.startswith("Custom"):
        custom_url = c1.text_input("URL or file path", value=local_clip)
    if source_choice.startswith("Local") and not have_clip:
        c1.warning("No local clip yet. Download one:\n\n"
                   "`python -m assbi.cli download --duration 120 --cookies-file cookies.txt`")
    endless = c2.checkbox(
        "♾️ Run until I press Stop", value=True,
        help="Keep monitoring the live feed forever; press Stop to finish & save.",
    )
    n_frames = c2.slider("Max frames (bounded mode)", 100, 8000, 2000, step=100,
                         disabled=endless)
    stride = c2.slider("Stride (every Nth frame)", 1, 10, 1,
                       help="Higher = cover more footage faster by skipping frames")
    session_name = c3.text_input("Session name", value="live")
    privacy_mode = c3.selectbox("🔒 Privacy", ["off", "blur", "pixelate"],
                                help="Anonymise people in the video (GDPR)")
    cookies_browser = c3.selectbox(
        "YouTube cookies", ["none", "chrome", "edge", "firefox", "brave"],
        help="Only for the live stream if you hit a bot check.",
    )

    if endless:
        st.caption("Endless live monitoring — runs until you press **Stop**, then "
                   "the session is saved. Local clip loops aren't endless (a file "
                   "ends); the live stream is.")
    else:
        coverage = n_frames * stride / 30.0
        st.caption(f"Will analyse ~{n_frames} frames (≈ {coverage:.0f}s of footage at "
                   f"stride {stride}), then stop automatically.")

    # Start/Stop control. Streamlit can't run a callback while the loop below is
    # blocking, so Stop works by triggering a *rerun*: the next frame's
    # ``video_slot.image()`` raises a RerunException that breaks the loop, and
    # the pipeline's ``finally`` still saves the session. The on_click callbacks
    # just flip the running flag so the rerun lands in the right branch.
    if "live_running" not in st.session_state:
        st.session_state.live_running = False

    bcols = st.columns(2)
    bcols[0].button("▶ Start live monitor", type="primary",
                    disabled=st.session_state.live_running,
                    on_click=lambda: st.session_state.update(live_running=True,
                                                             just_stopped=False))
    bcols[1].button("⏹ Stop", disabled=not st.session_state.live_running,
                    on_click=lambda: st.session_state.update(live_running=False,
                                                             just_stopped=True))

    if not st.session_state.live_running:
        if st.session_state.get("just_stopped"):
            st.success("⏹ Live monitor stopped — session saved. Select it in the "
                       "sidebar to explore the BI tabs.")
            st.session_state.just_stopped = False
        st.info("Pick a source and press **Start**. "
                + ("Press **Stop** any time to finish & save."
                   if endless else "It stops automatically after the chosen frames."))
        return

    if source_choice.startswith("Local"):
        source = local_clip
    elif source_choice.startswith("Assignment"):
        source = ASSIGNMENT_URL
    elif source_choice.startswith("Webcam"):
        source = 0
    else:
        source = custom_url.strip() or local_clip

    from assbi.pipeline.factory import build_pipeline, build_video_source, scale_lines_to_source
    from assbi.pipeline.orchestrator import LiveUpdate

    live_cfg = AppConfig.load("config/config.yaml")
    live_cfg.detection.backend = "yolo"
    live_cfg.video.max_frames = None if endless else n_frames  # None = endless
    live_cfg.video.stride = stride
    live_cfg.privacy.mode = privacy_mode
    if cookies_browser != "none":
        live_cfg.youtube.cookies_from_browser = cookies_browser

    video_slot = st.empty()
    mcols = st.columns(6)
    ph = [c.empty() for c in mcols]
    status = st.empty()
    progress = None if endless else st.progress(0.0, text="Connecting to source…")

    processed = {"n": 0}

    def on_frame(u: LiveUpdate):
        if u.image is not None:
            # OpenCV is BGR; Streamlit expects RGB. Update the image every few
            # frames at high speed to keep the browser responsive on big runs.
            # This st.image call is also where a pending Stop click interrupts
            # the loop (Streamlit raises a rerun here).
            if processed["n"] % (2 if stride == 1 else 1) == 0:
                video_slot.image(u.image[:, :, ::-1], channels="RGB", width="stretch")
        ph[0].metric("People IN", u.people_in)
        ph[1].metric("People OUT", u.people_out)
        ph[2].metric("Vehicles IN", u.vehicles_in)
        ph[3].metric("Vehicles OUT", u.vehicles_out)
        ph[4].metric("Live crowd", f"{u.person_count} ({u.density_level})")
        ph[5].metric("⚠️ Anomaly" if u.is_anomaly else "Status",
                     "SURGE" if u.is_anomaly else "normal")
        processed["n"] += 1
        if endless:
            status.caption(f"🔴 Live — {processed['n']} frames processed · "
                           f"press **Stop** to finish & save.")
        else:
            progress.progress(min(1.0, processed["n"] / n_frames),
                              text=f"Processing frame {processed['n']}/{n_frames}")
        # A clean stop can also break via the return value; a Stop click usually
        # interrupts earlier at the st.image call above.
        return st.session_state.get("live_running", True)

    try:
        src = build_video_source(live_cfg, source)
        scale_lines_to_source(live_cfg, src)   # land the line on any resolution
        pipeline = build_pipeline(live_cfg, repository=repo)
        with st.spinner("Loading YOLO model and opening the source…"):
            with src:
                result = pipeline.run(src, session_name, str(source), on_frame=on_frame)
    except Exception as exc:  # surface a clean message rather than a traceback
        # Let Streamlit's own control-flow exceptions (the rerun/stop raised when
        # the Stop button is clicked) propagate — the session was already saved
        # in the pipeline's ``finally``, and the rerun shows the idle/stopped UI.
        if exc.__class__.__name__ in {"RerunException", "StopException", "RerunData"}:
            raise
        if progress is not None:
            progress.empty()
        status.empty()
        st.session_state.live_running = False
        msg = str(exc)
        if "not a bot" in msg or "Sign in to confirm" in msg:
            st.error(
                "YouTube blocked the request with a bot check. Set **YouTube "
                "cookies** above to the browser you're logged into YouTube with, "
                "then press Start again."
            )
        elif "cookie database" in msg or "Could not copy" in msg:
            st.error(
                "Couldn't read the browser's cookies — it's locked while the "
                "browser is **open**. Fully close that browser and press Start "
                "again, or export a `cookies.txt` and set `youtube.cookies_file` "
                "in config.yaml."
            )
        elif "not found" in msg.lower() or "could not open" in msg.lower():
            st.error(
                f"Couldn't open the source. {msg}\n\nIf you picked the local clip, "
                "download one first:\n\n"
                "`python -m assbi.cli download --duration 120 --cookies-file cookies.txt`"
            )
        elif ("format is not available" in msg or "JavaScript runtime" in msg
              or "Only images" in msg or "n challenge" in msg):
            st.error(
                "YouTube needs a JavaScript runtime (the 'n-challenge'). Install "
                "**Deno** (`winget install DenoLand.Deno`), then restart this "
                "dashboard from a new terminal — the app auto-detects it."
            )
        else:
            st.error(f"Live run failed: {exc}")
        return

    # Reached only when the run ends on its own (bounded mode, or a file source
    # running out) — an endless live stop interrupts above and reruns instead.
    if progress is not None:
        progress.empty()
    status.empty()
    st.session_state.live_running = False
    s = result.summary
    st.success(
        f"✅ Done — {s.frames_processed} frames. "
        f"Vehicles {s.vehicles_in + s.vehicles_out} crossed "
        f"({s.vehicles_in} in / {s.vehicles_out} out); "
        f"people {s.people_in + s.people_out} ({s.people_in} in / {s.people_out} out); "
        f"peak crowd {s.peak_crowd}; {s.anomalies} anomalies."
    )
    st.caption(f"Saved as session **{session_name}** — select it in the sidebar to explore the BI tabs.")


def _render_export(config: AppConfig) -> None:
    st.divider()
    st.caption("📦 Export for Power BI / Excel (star-schema CSVs + Excel workbook + build guide).")
    if st.button("Export Power BI pack"):
        from assbi.bi_export import PowerBIExporter

        try:
            out, files = PowerBIExporter(config.database_path).export()
            st.success(f"Exported {len(files)} files to `{out}/`")
            st.code("\n".join(Path(f).name for f in files))
        except Exception as exc:
            st.error(f"Export failed: {exc}")


def _render_recording() -> None:
    path = Path("data/output/annotated.mp4")
    if not path.exists():
        return
    st.divider()
    st.subheader("🎞️ Last rendered clip")
    st.caption("Annotated output from the most recent `--render` run.")
    try:
        st.video(str(path))
    except Exception:
        st.info(f"Recorded clip at `{path}` (browser couldn't preview the codec).")


# --------------------------------------------------------------------------- #
#  BI views                                                                   #
# --------------------------------------------------------------------------- #
def _render_dataset(config: AppConfig, session_id: str) -> None:
    import sqlite3

    import pandas as pd

    st.subheader("📋 The analytics dataset")
    st.caption(
        "Structured data the pipeline derives from the unstructured video — the "
        "SQLite warehouse (`data/assbi.db`). This **is** the BI dataset: a star "
        "schema of one dimension and two fact tables."
    )

    conn = sqlite3.connect(config.database_path)
    try:
        counts = {
            t: pd.read_sql_query(f"SELECT COUNT(*) AS c FROM {t}", conn).iloc[0, 0]
            for t in ("sessions", "frame_analytics", "crossings")
        }
        frames = pd.read_sql_query(
            "SELECT * FROM frame_analytics WHERE session_id=? ORDER BY frame_index",
            conn, params=(session_id,))
        crossings = pd.read_sql_query(
            "SELECT * FROM crossings WHERE session_id=? ORDER BY frame_index",
            conn, params=(session_id,))
        dim = pd.read_sql_query("SELECT * FROM session_summary ORDER BY frames_processed DESC", conn)
    finally:
        conn.close()

    c = st.columns(3)
    c[0].metric("Sessions", f"{counts['sessions']:,}")
    c[1].metric("Frame records", f"{counts['frame_analytics']:,}")
    c[2].metric("Crossing events", f"{counts['crossings']:,}")
    st.markdown(
        "**Schema:** `dim_session` (1) → `fact_frame_analytics` (∗) & "
        "`fact_crossings` (∗), joined on `session_id`."
    )

    st.markdown(f"#### `fact_crossings` — line-crossing events for **{session_id}** ({len(crossings)} rows)")
    st.dataframe(crossings, width="stretch", height=240)
    st.download_button("⬇ Download crossings CSV", crossings.to_csv(index=False),
                       f"{session_id}_crossings.csv", "text/csv")

    st.markdown(f"#### `fact_frame_analytics` — per-frame time-series for **{session_id}** ({len(frames):,} rows)")
    st.dataframe(frames.head(1000), width="stretch", height=240)
    st.download_button("⬇ Download frames CSV", frames.to_csv(index=False),
                       f"{session_id}_frames.csv", "text/csv")

    st.markdown("#### `dim_session` — all analysis runs")
    st.dataframe(dim, width="stretch")
    st.caption("Full Power BI star-schema pack: **Overview tab → Export**, or "
               "`python -m assbi.cli powerbi`.")

    _render_training()


# Paths to the custom training dataset + fine-tune run (generated artifacts).
_DATASET_DIR = Path("data/dataset")
_RUN_DIR = Path("runs/assbi/finetune")
_MODEL_CHECK_DIR = Path("data/output/model_check")


@st.cache_data(show_spinner=False)
def _dataset_counts(dataset_dir: str) -> dict | None:
    d = Path(dataset_dir)
    if not d.exists():
        return None
    train = sorted((d / "images" / "train").glob("*.jpg"))
    val = sorted((d / "images" / "val").glob("*.jpg"))
    labels = 0
    for txt in (d / "labels").rglob("*.txt"):
        labels += sum(1 for line in txt.read_text(encoding="utf-8").splitlines() if line.strip())
    data_yaml = d / "data.yaml"
    return {
        "train": len(train),
        "val": len(val),
        "labels": labels,
        "data_yaml": data_yaml.read_text(encoding="utf-8") if data_yaml.exists() else "",
    }


@st.cache_data(show_spinner=False)
def _model_metrics(run_dir: str) -> dict | None:
    import pandas as pd

    csv = Path(run_dir) / "results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    last = df.iloc[-1]

    def g(key: str):
        return round(float(last[key]), 4) if key in df.columns else None

    return {
        "mAP50": g("metrics/mAP50(B)"),
        "mAP50-95": g("metrics/mAP50-95(B)"),
        "precision": g("metrics/precision(B)"),
        "recall": g("metrics/recall(B)"),
        "epochs": int(last["epoch"]) if "epoch" in df.columns else len(df),
    }


def _render_training() -> None:
    """Custom image dataset + fine-tuned model: counts, metrics and screenshots."""
    st.divider()
    st.subheader("🧠 Custom dataset & fine-tuned model")
    st.caption(
        "Beyond the analytics warehouse above, ASSBI also builds its **own image "
        "training dataset** from the footage and **fine-tunes YOLOv8n** on it — so "
        "the detector is trained on this exact camera, not just generic COCO."
    )

    ds = _dataset_counts(str(_DATASET_DIR))
    metrics = _model_metrics(str(_RUN_DIR))

    if ds is None and metrics is None:
        st.info(
            "No custom dataset or trained model found yet. Generate them with:\n\n"
            "```\npython -m assbi.cli build-dataset --source data/source_video.mp4 --frames 500\n"
            "python -m assbi.cli train --data data/dataset/data.yaml --epochs 30 --imgsz 416\n```"
        )
        return

    # -- dataset summary --------------------------------------------------
    if ds is not None:
        st.markdown("##### 📦 Training dataset (`data/dataset/`)")
        c = st.columns(4)
        c[0].metric("Train images", f"{ds['train']:,}")
        c[1].metric("Val images", f"{ds['val']:,}")
        c[2].metric("Person labels", f"{ds['labels']:,}")
        c[3].metric("Classes", "1 (person)")
        if ds["data_yaml"]:
            with st.expander("data.yaml (YOLO dataset descriptor)"):
                st.code(ds["data_yaml"], language="yaml")

    # -- model metrics ----------------------------------------------------
    if metrics is not None:
        st.markdown(f"##### 🎯 Fine-tuned model metrics ({metrics['epochs']} epochs)")
        m = st.columns(4)
        if metrics["mAP50"] is not None:
            m[0].metric("mAP@50", f"{metrics['mAP50']:.3f}")
        if metrics["mAP50-95"] is not None:
            m[1].metric("mAP@50-95", f"{metrics['mAP50-95']:.3f}")
        if metrics["precision"] is not None:
            m[2].metric("Precision", f"{metrics['precision']:.3f}")
        if metrics["recall"] is not None:
            m[3].metric("Recall", f"{metrics['recall']:.3f}")
        best = _RUN_DIR / "weights" / "best.pt"
        if best.exists():
            st.caption(f"Trained weights: `{best.as_posix()}` · metrics measured on the "
                       f"{ds['val'] if ds else 0}-image validation hold-out.")

    # -- screenshots gallery ---------------------------------------------
    gallery = [
        (_RUN_DIR / "val_batch0_pred.jpg", "Model predictions on validation frames"),
        (_RUN_DIR / "val_batch0_labels.jpg", "Ground-truth labels (same frames)"),
        (_RUN_DIR / "results.png", "Training curves (loss ↓, mAP ↑)"),
        (_RUN_DIR / "confusion_matrix.png", "Confusion matrix"),
        (_RUN_DIR / "BoxPR_curve.png", "Precision–Recall curve"),
        (_RUN_DIR / "train_batch0.jpg", "Labelled training batch"),
    ]
    present = [(p, cap) for p, cap in gallery if p.exists()]
    if present:
        st.markdown("##### 🖼️ Training evidence")
        cols = st.columns(3)
        for i, (p, cap) in enumerate(present):
            cols[i % 3].image(str(p), caption=cap, width="stretch")

    # -- live detections on real footage ---------------------------------
    checks = sorted(_MODEL_CHECK_DIR.glob("*.jpg")) if _MODEL_CHECK_DIR.exists() else []
    if checks:
        st.markdown("##### 👁️ The fine-tuned model on real footage")
        st.caption("Frames run through the trained model — boxes are live person detections.")
        cols = st.columns(min(4, len(checks)))
        for i, p in enumerate(checks):
            cols[i % len(cols)].image(str(p), caption=p.stem, width="stretch")

    _render_chatbot_nlu()


_INTENT_DIR = Path("models/intent")


@st.cache_data(show_spinner=False)
def _chatbot_metrics(intent_dir: str) -> dict | None:
    import json

    f = Path(intent_dir) / "metrics.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _render_chatbot_nlu() -> None:
    """Trained intent classifier (the chatbot's natural-language understanding)."""
    cm = _chatbot_metrics(str(_INTENT_DIR))
    if cm is None:
        return
    st.divider()
    st.subheader("💬 Chatbot NLU — trained intent classifier")
    approach = cm.get("approach", "neural net (bag-of-words → MLP)")
    st.caption(
        f"The chatbot's *understanding* is a model **trained** to map a question to "
        f"an intent — {approach}. The matched intent then pulls the real number from "
        "the warehouse, so answers stay grounded. (The reply wording can use the LLM; "
        "the **classification is trained**.)"
    )
    # Support both the transfer-learning and the bag-of-words metrics schemas.
    acc = cm.get("test_accuracy", cm.get("accuracy"))
    c = st.columns(4)
    c[0].metric("Test accuracy", f"{acc:.0%}" if acc is not None else "—")
    if cm.get("cv5_mean") is not None:
        c[1].metric("5-fold CV", f"{cm['cv5_mean']:.0%} ± {cm.get('cv5_std', 0):.0%}")
    else:
        c[1].metric("Macro F1", f"{cm['macro_f1']:.2f}")
    if cm.get("hard_eval_accuracy") is not None:
        c[2].metric("Hard novel set", f"{cm['hard_eval_accuracy']:.0%}",
                    help="Accuracy on hand-written paraphrases never seen in training")
    else:
        c[2].metric("Intents", len(cm.get("intents", [])))
    c[3].metric("Train / test", f"{cm['n_train']} / {cm['n_test']}")
    conf = _INTENT_DIR / "confusion_matrix.png"
    ds = _INTENT_DIR / "dataset.csv"
    cols = st.columns([3, 2])
    if conf.exists():
        cols[0].image(str(conf), caption="Intent confusion matrix (test set)", width="stretch")
    with cols[1]:
        st.markdown("**Recognised intents**")
        st.write(", ".join(cm.get("intents", [])))
        if ds.exists():
            import pandas as pd

            df = pd.read_csv(ds)
            st.caption(f"Training dataset: {len(df):,} labelled questions "
                       f"(`{ds.as_posix()}`)")
            st.download_button("⬇ Download intent dataset CSV", df.to_csv(index=False),
                               "intent_dataset.csv", "text/csv")


def _render_kpis(kpis: KPISet) -> None:
    c = st.columns(6)
    c[0].metric("People IN", kpis.total_people_in)
    c[1].metric("People OUT", kpis.total_people_out)
    c[2].metric("Net people", f"{kpis.net_people:+d}")
    c[3].metric("Vehicles IN", kpis.total_vehicles_in)
    c[4].metric("Vehicles OUT", kpis.total_vehicles_out)
    c[5].metric("Peak crowd", kpis.peak_crowd)
    c2 = st.columns(6)
    c2[0].metric("Total crossings", kpis.total_crossings)
    c2[1].metric("Anomalies", kpis.anomaly_count)
    c2[2].metric("Avg confidence", f"{kpis.avg_confidence:.0%}")
    c2[3].metric("Frames", kpis.frames_processed)


def _render_timeseries(series) -> None:
    st.subheader("Crowd & traffic over time")
    if not series:
        st.info("No frame data.")
        return
    import pandas as pd

    df = pd.DataFrame(
        {
            "frame": [f.frame_index for f in series],
            "people": [f.person_count for f in series],
            "vehicles": [f.vehicle_count for f in series],
        }
    ).set_index("frame")
    st.line_chart(df)


def _render_breakdown(builder: ReportBuilder, session_id: str) -> None:
    st.subheader("Line crossings by object class")
    bd = builder.class_breakdown(session_id)
    if not bd:
        st.info("No crossings recorded.")
        return
    import pandas as pd

    df = pd.DataFrame(bd).T.fillna(0).astype(int)
    df.columns = [c.upper() for c in df.columns]
    col1, col2 = st.columns([3, 2])
    col1.bar_chart(df)
    col2.dataframe(df, width="stretch")


def _render_forecast(builder: ReportBuilder, session_id: str) -> None:
    st.subheader("Predictive analytics — crowd forecast")
    f = builder.crowd_forecast(session_id)
    import pandas as pd

    col1, col2 = st.columns([2, 1])
    with col1:
        st.line_chart(pd.DataFrame({"forecast": f.predictions}))
    with col2:
        trend = "Rising 📈" if f.slope > 0.01 else "Falling 📉" if f.slope < -0.01 else "Stable ➡️"
        st.metric("Trend", trend, f"{f.slope:+.3f}/interval")
        st.metric("Model fit (R²)", f"{f.r_squared:.2f}")
        st.caption(f"Method: {f.method}")


def _render_anomalies(series) -> None:
    st.subheader("Anomaly detection")
    if not series:
        st.info("No frame data.")
        return
    flagged = [f for f in series if f.is_anomaly]
    st.metric("Anomalous frames", len(flagged))
    if not flagged:
        st.success("Crowd levels stayed within normal bounds — no anomalies flagged.")
        return
    import pandas as pd

    df = pd.DataFrame(
        {
            "frame": [f.frame_index for f in series],
            "crowd": [f.person_count for f in series],
            "anomaly": [f.person_count if f.is_anomaly else None for f in series],
        }
    ).set_index("frame")
    st.line_chart(df[["crowd"]])
    st.caption("Frames flagged as sudden surges or drops:")
    st.dataframe(
        pd.DataFrame(
            {
                "frame": [f.frame_index for f in flagged],
                "people": [f.person_count for f in flagged],
                "score": [round(f.anomaly_score, 2) for f in flagged],
            }
        ),
        width="stretch",
    )


def _render_assistant(config: AppConfig, repo, session_id: str) -> None:
    from assbi.pipeline.factory import build_assistant

    st.subheader("🤖 Ask the analytics assistant")
    assistant = build_assistant(config, repo, session_id)
    if assistant.llm is not None:
        st.caption(f"🟢 AI mode — grounded **{config.chatbot.model}** over this session's data.")
    else:
        st.caption(f"⚪ Rule-based mode. Set `${config.chatbot.api_key_env}` (then restart) for free-form AI chat.")

    if "history" not in st.session_state:
        st.session_state.history = []
    suggestions = ["Give me a summary", "How many cars crossed?", "Were there anomalies?", "What's the forecast?"]
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if cols[i].button(s):
            st.session_state.pending = s
    question = st.chat_input("Ask about people, vehicles, anomalies, forecast…")
    if "pending" in st.session_state:
        question = st.session_state.pop("pending")
    if question:
        ans = assistant.ask(question, history=st.session_state.history)
        st.session_state.history.append(("you", question))
        st.session_state.history.append(("bot", ans.text))
    for who, msg in st.session_state.history[-12:]:
        st.chat_message("user" if who == "you" else "assistant").write(msg)


if __name__ == "__main__":
    main()
