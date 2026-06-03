"""Power BI export pack.

Power BI ingests CSV/Excel natively, so rather than ship an opaque ``.pbix`` we
export the analytics warehouse as a clean **star schema** (one dimension, two
fact tables) plus a data dictionary and a step-by-step build guide with ready
DAX measures. This is the portable, reviewable way to hand a model to Power BI,
Tableau, Excel or any SQL warehouse.

    from assbi.bi_export import PowerBIExporter
    PowerBIExporter("data/assbi.db").export("data/output/powerbi")
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_DIM_SESSION_SQL = """
SELECT s.session_id,
       s.source,
       s.started_at,
       sm.frames_processed,
       sm.duration_seconds,
       sm.people_in, sm.people_out,
       sm.vehicles_in, sm.vehicles_out,
       sm.peak_crowd, sm.peak_crowd_frame,
       sm.anomalies, sm.avg_confidence
FROM sessions s
LEFT JOIN session_summary sm USING (session_id)
ORDER BY s.session_id
"""


class PowerBIExporter:
    """Dumps the SQLite warehouse to a Power BI-ready star-schema pack."""

    def __init__(self, db_path: str | Path = "data/assbi.db") -> None:
        self.db_path = Path(db_path)

    def export(self, out_dir: str | Path = "data/output/powerbi") -> tuple[Path, list[str]]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Warehouse not found: {self.db_path}. Run a session first.")
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        try:
            dim_session = pd.read_sql_query(_DIM_SESSION_SQL, conn)
            fact_frames = pd.read_sql_query(
                "SELECT * FROM frame_analytics ORDER BY session_id, frame_index", conn
            )
            fact_crossings = pd.read_sql_query(
                "SELECT * FROM crossings ORDER BY session_id, frame_index", conn
            )
        finally:
            conn.close()

        # A vehicle/person flag makes class-level DAX trivial in Power BI.
        if not fact_crossings.empty:
            vehicles = {"car", "truck", "bus", "motorcycle", "bicycle"}
            fact_crossings["category"] = fact_crossings["object_class"].apply(
                lambda c: "vehicle" if c in vehicles else "person"
            )

        written: list[str] = []
        tables = {
            "dim_session": dim_session,
            "fact_frame_analytics": fact_frames,
            "fact_crossings": fact_crossings,
        }
        for name, df in tables.items():
            p = out / f"{name}.csv"
            df.to_csv(p, index=False, encoding="utf-8-sig")  # BOM = clean Power BI import
            written.append(str(p))

        # Single Excel workbook (one sheet per table) — Power BI / Excel friendly.
        try:
            xlsx = out / "assbi_powerbi.xlsx"
            with pd.ExcelWriter(xlsx) as xw:
                for name, df in tables.items():
                    df.to_excel(xw, sheet_name=name[:31], index=False)
            written.append(str(xlsx))
        except Exception:  # openpyxl missing — CSVs are enough
            pass

        (out / "DATA_DICTIONARY.md").write_text(_DATA_DICTIONARY, encoding="utf-8")
        (out / "POWERBI_SETUP.md").write_text(_POWERBI_SETUP, encoding="utf-8")
        written.append(str(out / "DATA_DICTIONARY.md"))
        written.append(str(out / "POWERBI_SETUP.md"))
        return out, written


_DATA_DICTIONARY = """# ASSBI — Power BI Data Dictionary

Star schema: **`dim_session`** (dimension) joins to two fact tables on
`session_id`.

```
                +------------------+
                |   dim_session    |   (1)
                |  session_id (PK) |
                +------------------+
                   |            |
            (*)    |            |   (*)
   +-------------------+   +------------------+
   | fact_frame_analytics|  |  fact_crossings  |
   |  session_id (FK)   |   |  session_id (FK) |
   +-------------------+   +------------------+
```

## dim_session  — one row per analysis run
| Column | Type | Meaning |
| --- | --- | --- |
| session_id | text (PK) | Unique run id |
| source | text | Video file / stream URL |
| started_at | datetime | UTC start time |
| frames_processed | int | Frames analysed |
| duration_seconds | number | Footage length |
| people_in / people_out | int | Pedestrian line crossings by direction |
| vehicles_in / vehicles_out | int | Vehicle line crossings by direction |
| peak_crowd | int | Max simultaneous people |
| peak_crowd_frame | int | Frame of the peak |
| anomalies | int | Anomalous frames flagged |
| avg_confidence | number | Mean detection confidence (0-1) |

## fact_frame_analytics — one row per frame (time-series grain)
| Column | Type | Meaning |
| --- | --- | --- |
| session_id | text (FK) | → dim_session |
| frame_index | int | Frame number |
| ts | datetime | Frame timestamp (UTC) |
| person_count / vehicle_count | int | Objects in view that frame |
| total | int | Total detections |
| crossings_in / crossings_out | int | Cumulative crossings so far |
| is_anomaly | int (0/1) | Frame flagged anomalous |
| anomaly_score | number | Robust z-score magnitude |

## fact_crossings — one row per line-crossing event
| Column | Type | Meaning |
| --- | --- | --- |
| session_id | text (FK) | → dim_session |
| track_id | int | Tracked object id |
| object_class | text | person / car / truck / bus / motorcycle / bicycle |
| category | text | person or vehicle (derived) |
| direction | text | in / out |
| line_name | text | Which counting line |
| frame_index | int | Frame of the crossing |
| ts | datetime | Event timestamp (UTC) |

Data is **structured** (these tables) derived from **unstructured** video —
the structured/unstructured bridge the brief asks for.
"""


_POWERBI_SETUP = """# Building the ASSBI dashboard in Power BI Desktop

1. **Get Data → Text/CSV** (or **Excel** → `assbi_powerbi.xlsx`) and load all
   three tables: `dim_session`, `fact_frame_analytics`, `fact_crossings`.
2. **Model view → create relationships** (drag `session_id`):
   - `dim_session[session_id]`  →  `fact_frame_analytics[session_id]`  (1 : *)
   - `dim_session[session_id]`  →  `fact_crossings[session_id]`         (1 : *)
   Set cross-filter direction to *Single* (from the dimension).
3. **New measures** (paste into the Modeling tab):

```DAX
Total Crossings   = COUNTROWS(fact_crossings)
People Crossed    = CALCULATE([Total Crossings], fact_crossings[category] = "person")
Vehicles Crossed  = CALCULATE([Total Crossings], fact_crossings[category] = "vehicle")
Crossings In      = CALCULATE([Total Crossings], fact_crossings[direction] = "in")
Crossings Out     = CALCULATE([Total Crossings], fact_crossings[direction] = "out")
Peak Crowd        = MAX(fact_frame_analytics[person_count])
Anomaly Frames    = CALCULATE(COUNTROWS(fact_frame_analytics), fact_frame_analytics[is_anomaly] = 1)
Avg Confidence    = AVERAGE(dim_session[avg_confidence])
```

4. **Suggested visuals**
   - **Cards:** People Crossed, Vehicles Crossed, Peak Crowd, Anomaly Frames.
   - **Line chart:** axis `fact_frame_analytics[frame_index]`, values
     `person_count` & `vehicle_count` — crowd/traffic over time.
   - **Clustered bar:** axis `object_class`, value Total Crossings, legend
     `direction` — the line-crossing breakdown.
   - **Stacked column:** axis `dim_session[session_id]`, values People/Vehicles
     Crossed — compare sessions.
   - **Table:** anomalous frames (filter `is_anomaly = 1`) with `anomaly_score`.
5. **Slicer:** `dim_session[session_id]` to switch between runs.

Re-export any time after new runs with: `python -m assbi.cli powerbi`.
"""
