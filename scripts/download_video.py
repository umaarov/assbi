"""Convenience script to fetch the assignment YouTube video.

Usage:
    python scripts/download_video.py [URL]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assbi.video.youtube import DEFAULT_URL, download_video  # noqa: E402

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    path = download_video(url)
    print(f"Saved to {path}")
