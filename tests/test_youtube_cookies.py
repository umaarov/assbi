"""The YouTube cookie options must reach yt-dlp (offline, mocked)."""
import sys
import types


def _install_fake_ytdlp(monkeypatch, captured: dict):
    class FakeYDL:
        def __init__(self, opts):
            captured["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {"url": "http://cdn/stream.m3u8"}

    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)


def test_stream_url_prefers_cookie_file(monkeypatch, tmp_path):
    captured: dict = {}
    _install_fake_ytdlp(monkeypatch, captured)
    from assbi.video.youtube import stream_url

    cookie = tmp_path / "c.txt"
    cookie.write_text("# Netscape HTTP Cookie File\n")
    out = stream_url("https://youtu.be/x", cookies_from_browser="chrome", cookies_file=str(cookie))
    assert out == "http://cdn/stream.m3u8"
    # An existing cookie file wins over the (lockable) browser DB.
    assert captured["opts"]["cookiefile"] == str(cookie)
    assert "cookiesfrombrowser" not in captured["opts"]


def test_stream_url_falls_back_to_browser_when_no_file(monkeypatch):
    captured: dict = {}
    _install_fake_ytdlp(monkeypatch, captured)
    from assbi.video.youtube import stream_url

    # cookies_file path doesn't exist -> use the browser instead.
    stream_url("https://youtu.be/x", cookies_from_browser="edge", cookies_file="nope.txt")
    assert captured["opts"]["cookiesfrombrowser"] == ("edge",)
    assert "cookiefile" not in captured["opts"]


def test_stream_url_without_cookies_omits_them(monkeypatch):
    captured: dict = {}
    _install_fake_ytdlp(monkeypatch, captured)
    from assbi.video.youtube import stream_url

    stream_url("https://youtu.be/x")
    assert "cookiesfrombrowser" not in captured["opts"]
    assert "cookiefile" not in captured["opts"]
