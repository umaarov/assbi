# ASSBI — AI-Powered Smart Surveillance & Business Intelligence Platform

A computer-vision **Business Intelligence** system that monitors human and
vehicle movement in video, **counts people and cars crossing a virtual line**,
measures crowd density, flags anomalies, forecasts footfall, persists everything
to an analytics warehouse, and surfaces it through a **BI dashboard** and an
**AI chatbot**.

Built with a clean, layered (hexagonal) architecture so the heavy CV stack
(YOLO + OpenCV) is a *pluggable adapter* — the entire pipeline also runs on a
**synthetic detector with zero third-party dependencies**, so you can
demonstrate the full system on any machine.

---

## Architecture

```
            ┌─────────────────────── presentation ───────────────────────┐
            │   CLI (assbi.cli)   ·   Streamlit dashboard   ·   Chatbot    │
            └───────────────▲───────────────────────────────▲─────────────┘
                            │                                │
            ┌───────────────┴──────── application ───────────┴─────────────┐
            │  pipeline.orchestrator  ·  pipeline.factory  ·  reporting     │
            └───────────────▲───────────────────────────────▲──────────────┘
                            │   depends only on ports (ABCs) │
            ┌───────────────┴────────────  domain  ──────────┴─────────────┐
            │   geometry · models · interfaces (ObjectDetector,            │
            │   VideoSource, AnalyticsRepository)  — framework-free          │
            └───────────────▲───────────────────────────────▲──────────────┘
                            │   adapters implement the ports │
            ┌───────────────┴──────── infrastructure ────────┴─────────────┐
            │ detection(YOLO│sim) · tracking · video(OpenCV│sim│YT) ·       │
            │ analytics(line/crowd/anomaly/forecast) · persistence(SQLite)  │
            └──────────────────────────────────────────────────────────────┘
```

Layer rule: dependencies point **inward**. The domain knows nothing about YOLO,
OpenCV, SQLite or Streamlit; adapters depend on the domain, never the reverse.

```
src/assbi/
├── domain/        geometry, entities, ports (pure stdlib, 100% unit-tested)
├── detection/     YOLODetector (ultralytics) + SimulationDetector (synthetic)
├── tracking/      CentroidTracker (IoU + centroid association)
├── analytics/     line_counter, crowd, anomaly, prediction
├── persistence/   SQLiteAnalyticsRepository (star-schema warehouse)
├── video/         OpenCV source, synthetic source, YouTube fetch
├── pipeline/      orchestrator (use-case) + factory (composition root)
├── reporting/     KPIs, CSV/JSON/Markdown exports, forecasts
├── chatbot/       natural-language analytics assistant (+ optional LLM hook)
├── dashboard/     Streamlit BI dashboard
└── cli.py         command-line entry point
```

---

## Quick start (no installs required)

The synthetic pipeline runs on the **Python standard library alone**:

```bash
# from the project root
set PYTHONPATH=src           # Windows (PowerShell: $env:PYTHONPATH="src")
# export PYTHONPATH=src      # macOS/Linux

python scripts/run_demo.py                       # full E2E demo + report + chatbot
python -m assbi.cli run --session demo            # run an analysis session
python -m assbi.cli report --session demo --export
python -m assbi.cli chat --session demo           # interactive Q&A
```

### AI chatbot

The assistant is a **real LLM** (DeepSeek, OpenAI-compatible) *grounded* in the
session's analytics — it answers free-form questions, greetings and follow-ups
while taking every number from the warehouse, so it can't hallucinate counts.
Set the key (read from an env var, never hardcoded) and it switches on
automatically; without a key it falls back to a deterministic rule engine.

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."        # current shell;  setx … for permanent
python -m assbi.cli chat --session pattaya
```

```
you> Hello
bot> Hi! I'm ASSBI — I can report on people & vehicle counts, crossings,
     peak crowd, anomalies and the crowd forecast. What would you like to know?
you> how many cars crossed and was it busy?
bot> 23 cars crossed in total (16 in, 7 out). Peak crowd was 12; 1 anomaly was
     flagged — so it was moderately busy.
```

Provider/model are configurable in `config.yaml` under `chatbot:` (any
OpenAI-compatible endpoint works by changing `base_url`/`model`).

## Running on the real YouTube video (full CV stack)

The line-counting requirement (people & cars crossing a line on
`https://www.youtube.com/watch?v=7uG-gbg0I8Y`) runs with the YOLO backend.
Install the stack once:

```bash
pip install -r requirements-full.txt     # ultralytics, opencv, torch, yt-dlp …
```

**Real-time, no download (recommended)** — resolve the stream to a direct media
URL and process frames as they arrive:

```bash
python -m assbi.cli run --stream --render --frames 1800 --session pattaya
python -m assbi.cli report --session pattaya
python -m assbi.cli dashboard            # launch the Streamlit BI dashboard
```

`--stream` implies `--backend yolo`; `--frames` bounds the run (the feed is a
~5.5 h live cam). You can also stream any URL or a webcam:

```bash
python -m assbi.cli run --source "https://youtu.be/7uG-gbg0I8Y" --backend yolo
python -m assbi.cli run --source 0 --backend yolo          # local webcam
```

> **YouTube streaming prerequisites & troubleshooting.** YouTube now requires a
> **JavaScript runtime** to decode stream URLs (the "n-challenge"). Install
> **Deno** once — the app auto-detects it:
> ```bash
> winget install DenoLand.Deno      # then open a new terminal
> ```
> Symptoms without it: *"Requested format is not available" / "Only images are
> available"*.
>
> **"Sign in to confirm you're not a bot"?** YouTube sometimes also rate-limits
> unauthenticated requests. Often transient (just retry), otherwise pass your
> browser cookies:
> ```bash
> python -m assbi.cli run --stream --cookies-from-browser edge   # close the browser first
> # or, without closing it — export youtube.com cookies with a browser extension:
> python -m assbi.cli run --stream --cookies-file cookies.txt
> ```
> The dashboard's Live Monitor has a **YouTube cookies** selector for the same.

**Offline (download first)** — caches the file, then analyses it:

```bash
python -m assbi.cli download                               # → data/source_video.mp4
python -m assbi.cli run --source data/source_video.mp4 --backend yolo --render
```

`--render` writes an annotated MP4 (`data/output/annotated.mp4`) with the
counting line, tracked boxes, motion trails and live IN/OUT counters drawn on.

**Privacy / GDPR** — anonymise people in the output (identities hidden, vehicles
and analytics untouched):

```bash
python -m assbi.cli run --stream --render --privacy blur     # or: pixelate
```

> **Python version:** verified end-to-end on **Python 3.14** (torch 2.12 +cpu,
> opencv 4.13, ultralytics 8.4) as well as 3.11/3.12. If your interpreter has no
> CV wheels yet, the dependency-free synthetic demo still runs everywhere.

## Power BI / Excel export

Export the warehouse as a **star-schema pack** Power BI ingests directly:

```bash
python -m assbi.cli powerbi          # → data/output/powerbi/
```

Produces `dim_session.csv`, `fact_frame_analytics.csv`, `fact_crossings.csv`, an
`assbi_powerbi.xlsx` workbook, a **DATA_DICTIONARY.md**, and **POWERBI_SETUP.md**
(relationships + ready-to-paste DAX measures). You can also click *Export Power
BI pack* in the dashboard's Overview tab.

## BI dashboard

```bash
python -m assbi.cli dashboard              # or: streamlit run src/assbi/dashboard/app.py
```

A single command-center, organised into tabs:

| Tab | Contents |
| --- | -------- |
| 🔴 **Live Monitor** | Runs YOLO on the live stream (or a custom URL / webcam) and renders the annotated video **in real time** with live IN/OUT counters and a **privacy** (blur/pixelate) toggle. Saves the session to the warehouse. |
| 📊 Overview | KPI cards, **Power BI export** button, and the last rendered annotated clip. |
| 📈 Trends | Crowd & vehicle time-series. |
| 🚦 Crossings | Per-class line-crossing breakdown (chart + table). |
| 🔮 Forecast | Predictive crowd forecast (trend, R², projection). |
| ⚠️ Anomalies | Flagged surge/drop frames with scores. |
| 🤖 Assistant | The embedded natural-language analytics chatbot. |

The Live Monitor needs the full CV stack (`requirements-full.txt`); the other
tabs only need `requirements.txt` (pyyaml, pandas, streamlit, requests). The
deployed cloud dashboard uses the lightweight `requirements.txt`.

---

## Configuration

Everything is driven by `config/config.yaml` (counting lines, detection
backend, tracker/crowd/anomaly thresholds, video settings). All values have
in-code defaults, so the file is optional. Move the lines, add more, or switch
`detection.backend` between `simulation` and `yolo` without touching code.

## Tests

```bash
pip install pytest
set PYTHONPATH=src
python -m pytest        # 23 tests: geometry, line counting, tracking,
                        # analytics, and a full pipeline→warehouse→chatbot E2E
```

---

## How the build maps to the assignment tasks

| Brief requirement | Where it lives |
| --- | --- |
| YOLO object detection | `detection/yolo_detector.py` |
| OpenCV video analytics | `video/opencv_source.py`, `pipeline/annotator.py` |
| **Line-crossing people & car counts** | `analytics/line_counter.py`, `domain/geometry.py` |
| Human detection & crowd counting | `analytics/crowd.py` |
| Object tracking | `tracking/centroid_tracker.py` |
| Anomaly detection | `analytics/anomaly.py` (rolling robust z-score) |
| Predictive analytics | `analytics/prediction.py` (linear + Holt) |
| Data pipeline (collect→store→process→serve) | `pipeline/orchestrator.py`, `persistence/` |
| Data storage / modelling (SQL warehouse) | `persistence/sqlite_repository.py` |
| Dashboards, KPIs, charts, reports | `dashboard/app.py`, `reporting.py` |
| AI chatbot assistant | `chatbot/assistant.py` |
| Real-time monitoring | streaming frame loop in the orchestrator |
