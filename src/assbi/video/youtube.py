"""Helpers to obtain the assignment's YouTube video for analysis.

Two modes, both delegated to ``yt-dlp`` (an optional dependency):

* :func:`download_video` caches the file locally for offline processing.
* :func:`stream_url` resolves the page to a direct CDN media URL **without
  downloading**, so frames can be pulled in real time. The user supplies the
  URL; we only read it for processing.
"""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

DEFAULT_URL = "https://www.youtube.com/watch?v=3nyPER2kzqk"  # EarthCam Live: Dublin, Ireland


def ensure_js_runtime() -> bool:
    """Make sure a JS runtime (Deno) is on PATH for YouTube's n-challenge.

    YouTube now requires a JS runtime to decode stream URLs; yt-dlp uses Deno.
    winget installs it but only updates PATH for *new* shells, so we proactively
    locate the binary in common install dirs and prepend it. Returns True if a
    runtime is available.
    """
    if shutil.which("deno"):
        return True
    candidates: list[str] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates += glob.glob(
            os.path.join(local, "Microsoft", "WinGet", "Packages", "DenoLand.Deno*", "deno.exe")
        )
        candidates.append(os.path.join(local, "Deno", "deno.exe"))
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    candidates.append(os.path.join(home, ".deno", "bin", "deno.exe"))
    for c in candidates:
        if os.path.isfile(c):
            os.environ["PATH"] = os.path.dirname(c) + os.pathsep + os.environ.get("PATH", "")
            return True
    return False


def find_ffmpeg() -> str | None:
    """Locate ffmpeg.exe (needed to download a bounded clip). Returns its dir."""
    exe = shutil.which("ffmpeg")
    if exe:
        return os.path.dirname(exe)
    cands: list[str] = [
        os.path.join(os.getcwd(), "tools", "ffmpeg.exe"),  # project-local build
    ]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands += glob.glob(
            os.path.join(local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*",
                         "**", "ffmpeg.exe"),
            recursive=True,
        )
    for c in cands:
        if os.path.isfile(c):
            return os.path.dirname(c)
    return None


def is_youtube_url(source: object) -> bool:
    """True if ``source`` looks like a YouTube watch/share URL."""
    return isinstance(source, str) and (
        "youtube.com/" in source or "youtu.be/" in source
    )


def _cookie_opts(cookies_from_browser: str | None, cookies_file: str | None) -> dict:
    """yt-dlp options to authenticate with cookies.

    YouTube increasingly returns "Sign in to confirm you're not a bot" for
    unauthenticated requests; passing cookies fixes it. A cookies *file* is
    preferred when present — it works even while the browser is open, whereas
    reading the live browser DB fails because Chromium locks it.
    """
    opts: dict = {}
    if cookies_file and os.path.isfile(cookies_file):
        opts["cookiefile"] = cookies_file
    elif cookies_from_browser:
        # yt-dlp expects a tuple: (browser[, profile, keyring, container]).
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def stream_url(
    url: str = DEFAULT_URL,
    max_height: int = 720,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    remote_components: tuple[str, ...] | list[str] | None = ("ejs:github",),
) -> str:
    """Resolve ``url`` to a direct media URL playable by OpenCV — no download.

    Picks a single *progressive* stream (combined video+audio, or video-only)
    at or below ``max_height`` so ``cv2.VideoCapture`` can open it directly.
    Requires ``pip install yt-dlp``. Pass ``cookies_from_browser`` (e.g.
    ``"chrome"``) or ``cookies_file`` to get past YouTube's bot check.
    """
    try:
        import yt_dlp  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "yt-dlp is not installed. Run `pip install yt-dlp` to stream videos."
        ) from exc

    ensure_js_runtime()

    # Prefer a single progressive mp4 (has both A/V in one URL); fall back to
    # the best video-only stream OpenCV can still decode.
    ydl_opts = {
        "format": (
            f"best[height<={max_height}][ext=mp4][vcodec!=none][acodec!=none]"
            f"/best[height<={max_height}][vcodec!=none][acodec!=none]"
            f"/bestvideo[height<={max_height}][ext=mp4]/best"
        ),
        "quiet": True,
        "noplaylist": True,
        **_cookie_opts(cookies_from_browser, cookies_file),
    }
    if remote_components:
        # Allow yt-dlp to fetch the EJS challenge-solver scripts (needs Deno) so
        # YouTube's n-challenge can be solved and real video formats appear.
        ydl_opts["remote_components"] = list(remote_components)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # pragma: no cover - network I/O
        info = ydl.extract_info(url, download=False)
        direct = info.get("url")
        if direct:
            return direct
        formats = info.get("formats") or []
        if formats:
            return formats[-1]["url"]
    raise RuntimeError(f"Could not resolve a streamable URL for {url!r}")


def download_video(
    url: str = DEFAULT_URL,
    output_dir: str | Path = "data",
    filename: str = "source_video.mp4",
    max_height: int = 720,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    remote_components: tuple[str, ...] | list[str] | None = ("ejs:github",),
    duration: float | None = None,
) -> Path:
    """Download ``url`` to ``output_dir/filename`` and return the path.

    Requires ``pip install yt-dlp``. Caps resolution at ``max_height`` to keep
    inference real-time-ish on CPU. Pass ``cookies_from_browser``/``cookies_file``
    to authenticate past YouTube's bot check. ``duration`` (seconds) downloads
    only the first N seconds — ideal for the 5.5 h live cam (needs ffmpeg). A
    local clip analyses at full speed, immune to YouTube's stream throttling.
    """
    try:
        import yt_dlp  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "yt-dlp is not installed. Run `pip install yt-dlp` to download videos."
        ) from exc

    ensure_js_runtime()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename

    # Prefer a single progressive stream so no ffmpeg merge is needed. Only fall
    # back to a separate video+audio merge (which requires ffmpeg) as a last
    # resort. We don't need audio for analysis anyway.
    ydl_opts = {
        "format": (
            f"best[height<={max_height}][ext=mp4][vcodec!=none][acodec!=none]"
            f"/best[height<={max_height}][vcodec!=none][acodec!=none]"
            f"/bestvideo[height<={max_height}][ext=mp4]+bestaudio"
            f"/best[height<={max_height}]"
        ),
        "outtmpl": str(target),
        "quiet": False,
        "noplaylist": True,
        **_cookie_opts(cookies_from_browser, cookies_file),
    }
    if remote_components:
        ydl_opts["remote_components"] = list(remote_components)

    ffmpeg_dir = find_ffmpeg()
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir
    if duration:
        if not ffmpeg_dir:
            raise RuntimeError(
                "Downloading a bounded clip needs ffmpeg. Install it with "
                "`winget install Gyan.FFmpeg` (open a new terminal after)."
            )
        from yt_dlp.utils import download_range_func

        ydl_opts["download_ranges"] = download_range_func(None, [(0.0, float(duration))])
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # pragma: no cover - network I/O
        ydl.download([url])
    return target
