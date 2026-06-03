"""Typed application configuration with optional YAML overrides.

Defaults make the synthetic demo work out-of-the-box. A ``config.yaml`` (or any
path passed to :meth:`AppConfig.load`) can override any field. PyYAML is
optional — without it, defaults (and JSON files) still work.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """Load ``KEY=value`` lines from a ``.env`` file into ``os.environ``.

    A real exported environment variable always wins (we only fill what's
    missing), so this is a convenience for storing secrets like the DeepSeek API
    key in a file instead of the shell. Lines starting with ``#`` are ignored.
    No third-party dependency required.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass
class LineConfig:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass
class DetectionConfig:
    backend: str = "simulation"          # "simulation" | "yolo"
    model_path: str = "yolov8n.pt"
    confidence: float = 0.35
    iou: float = 0.5
    device: str | None = None


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_missed: int = 30
    max_distance: float = 80.0


@dataclass
class CrowdConfig:
    moderate: int = 8
    high: int = 20
    critical: int = 40


@dataclass
class AnomalyConfig:
    window: int = 90
    threshold: float = 4.0
    warmup: int = 30
    min_scale: float = 1.0


@dataclass
class YouTubeConfig:
    """Auth/runtime for resolving YouTube streams past anti-automation."""
    cookies_from_browser: str | None = None   # "chrome" | "edge" | "firefox" | ...
    cookies_file: str | None = None           # path to a cookies.txt export
    # Remote solver scripts for the n-challenge (needs a JS runtime / Deno).
    remote_components: list[str] = field(default_factory=lambda: ["ejs:github"])


@dataclass
class PrivacyConfig:
    """Anonymisation applied to the annotated video for GDPR-style privacy."""
    mode: str = "off"                    # "off" | "blur" | "pixelate"
    targets: list[str] = field(default_factory=lambda: ["person"])
    strength: int = 23                   # blur kernel / pixelation coarseness (px)


@dataclass
class ChatbotConfig:
    provider: str = "deepseek"           # "deepseek"/"openai-compatible" | "none"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"  # env var holding the key (never inline)
    temperature: float = 0.2


@dataclass
class VideoConfig:
    width: int = 1280
    height: int = 720
    fps: float = 25.0
    total_frames: int = 500              # used by the simulation source
    stride: int = 1
    max_frames: int | None = None
    render: bool = False                 # write an annotated output video
    output_path: str = "data/output/annotated.mp4"


@dataclass
class AppConfig:
    database_path: str = "data/assbi.db"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    crowd: CrowdConfig = field(default_factory=CrowdConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    chatbot: ChatbotConfig = field(default_factory=ChatbotConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    lines: list[LineConfig] = field(
        default_factory=lambda: [
            # Horizontal mid-line: counts pedestrians moving up/down.
            LineConfig("pedestrian_line", (0, 360), (1280, 360)),
            # Lower horizontal line across the road: counts vehicles.
            LineConfig("vehicle_line", (0, 560), (1280, 560)),
        ]
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            return cfg
        raw = _read_structured(p)
        return cfg._merge(raw)

    def _merge(self, raw: dict) -> "AppConfig":
        if not raw:
            return self
        if "database_path" in raw:
            self.database_path = raw["database_path"]
        self.detection = _merge_dc(self.detection, raw.get("detection"))
        self.tracking = _merge_dc(self.tracking, raw.get("tracking"))
        self.crowd = _merge_dc(self.crowd, raw.get("crowd"))
        self.anomaly = _merge_dc(self.anomaly, raw.get("anomaly"))
        self.chatbot = _merge_dc(self.chatbot, raw.get("chatbot"))
        self.privacy = _merge_dc(self.privacy, raw.get("privacy"))
        self.youtube = _merge_dc(self.youtube, raw.get("youtube"))
        self.video = _merge_dc(self.video, raw.get("video"))
        if "lines" in raw and raw["lines"]:
            self.lines = [
                LineConfig(item["name"], tuple(item["start"]), tuple(item["end"]))
                for item in raw["lines"]
            ]
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_dc(instance, overrides: dict | None):
    if not overrides:
        return instance
    for key, value in overrides.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def _read_structured(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: WPS433

            return yaml.safe_load(text) or {}
        except ImportError:
            # Best-effort: a flat YAML may still be JSON-compatible.
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # PyYAML absent and not JSON: fall back to in-code defaults so
                # the platform stays runnable with zero third-party deps.
                import warnings

                warnings.warn(
                    f"PyYAML not installed; ignoring '{path}' and using built-in "
                    "defaults. Run `pip install pyyaml` to honour the YAML config.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return {}
    return json.loads(text)
