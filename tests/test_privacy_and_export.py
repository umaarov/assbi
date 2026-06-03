"""Tests for the privacy anonymiser and the Power BI export pack."""
import tempfile
from pathlib import Path

import pytest

from assbi.config import AppConfig
from assbi.domain.geometry import BoundingBox, Point
from assbi.domain.models import ObjectClass, Track
from assbi.persistence.sqlite_repository import SQLiteAnalyticsRepository
from assbi.pipeline.annotator import FrameAnnotator
from assbi.pipeline.factory import build_pipeline, build_video_source


def _track(cls: ObjectClass) -> Track:
    return Track(1, cls, BoundingBox(20, 20, 60, 60), 0.9, Point(40, 40), 0)


def _noisy_image():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    img = np.full((100, 100, 3), 127, dtype=np.uint8)
    img[20:60, 20:60] = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    return np, img


@pytest.mark.parametrize("mode", ["blur", "pixelate"])
def test_privacy_anonymises_person_region(mode):
    np, img = _noisy_image()
    ann = FrameAnnotator([], privacy_mode=mode)
    if not ann.available:
        pytest.skip("OpenCV not installed")
    before = img[20:60, 20:60].copy()
    ann.anonymize(img, [_track(ObjectClass.PERSON)])
    assert not np.array_equal(img[20:60, 20:60], before), "person region must be obscured"


def test_privacy_leaves_non_targets_untouched():
    np, img = _noisy_image()
    ann = FrameAnnotator([], privacy_mode="blur")  # default targets = ["person"]
    if not ann.available:
        pytest.skip("OpenCV not installed")
    before = img[20:60, 20:60].copy()
    ann.anonymize(img, [_track(ObjectClass.CAR)])
    assert np.array_equal(img[20:60, 20:60], before), "a car must not be anonymised"


def test_powerbi_export_produces_star_schema():
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.database_path = str(Path(tmp) / "wh.db")
        cfg.video.total_frames = 150
        repo = SQLiteAnalyticsRepository(cfg.database_path)
        pipeline = build_pipeline(cfg, repository=repo)
        src = build_video_source(cfg, None)
        with src:
            pipeline.run(src, "s1", "simulation:test")
        repo.close()

        from assbi.bi_export import PowerBIExporter

        out, files = PowerBIExporter(cfg.database_path).export(Path(tmp) / "powerbi")
        names = {Path(f).name for f in files}
        assert {"dim_session.csv", "fact_frame_analytics.csv", "fact_crossings.csv"} <= names
        assert (out / "POWERBI_SETUP.md").exists()
        assert (out / "DATA_DICTIONARY.md").exists()

        dim = pd.read_csv(out / "dim_session.csv")
        assert len(dim) == 1 and dim.iloc[0]["session_id"] == "s1"
        crossings = pd.read_csv(out / "fact_crossings.csv")
        # derived category column drives class-level DAX
        assert "category" in crossings.columns
        assert set(crossings["category"].unique()) <= {"person", "vehicle"}
