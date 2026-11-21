"""Editable phrases and the first-run model download (no network: a fake download is used)."""
import io

import pytest

from nullpoint.board import phrases as ph
from nullpoint.capture import ensure_model


def test_phrases_validate_save_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "CUSTOM", tmp_path / "phrases.json")
    try:
        ph.set_pages([["Mine", ["Coffee, please", "  ", "Open the window", ph.NEXT]], ["Family", ["Call Mum"]]])
        assert ph.PAGES[0] == ("Mine", ["Coffee, please", "Open the window", ph.NEXT])    # blanks dropped, NEXT once
        assert ph.load() == ph.PAGES and ph.editable()[1] == ["Family", ["Call Mum"]]
        for bad in ([], [["x", []]], [["x", ["a" * 41]]], [["x", ["a"] * 8]], "nope"):
            with pytest.raises(ValueError):
                ph.set_pages(bad)
        assert ph.PAGES[0][0] == "Mine"                          # a bad save leaves the board unchanged
    finally:
        ph.set_pages(ph.DEFAULT_PAGES, save=False)


def test_model_downloads_once_and_cleans_up_on_failure(tmp_path):
    target = tmp_path / "models" / "face.task"

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    ensure_model(target, urlopen=lambda url, timeout: Resp(b"x" * 2_000_000))
    assert target.stat().st_size == 2_000_000
    assert ensure_model(target, urlopen=lambda *a, **k: 1 / 0) == target       # already there: no download

    other = tmp_path / "models" / "other.task"
    with pytest.raises(SystemExit):
        ensure_model(other, urlopen=lambda url, timeout: Resp(b"<html>error</html>"))
    assert not other.exists() and not other.with_suffix(".part").exists()


def test_phrase_editing_over_wifi_needs_the_pin(tmp_path, monkeypatch):
    import argparse
    import asyncio
    import json
    from types import SimpleNamespace
    from nullpoint.board import server
    monkeypatch.setattr(ph, "CUSTOM", tmp_path / "phrases.json")
    args = argparse.Namespace(sim=True, no_adapt=False, lan=True, mute=True, no_setup=True, cam=0, port=8765,
                              profile=tmp_path / "p.json", no_browser=True, video=None)
    app = server.build_app(args)
    post = [r for r in app.routes if getattr(r, "path", "") == "/phrases" and "POST" in r.methods][0].endpoint
    pin = [r for r in app.routes if getattr(r, "path", "") == "/pin"][0].endpoint

    class Req:
        def __init__(self, host, body): self.client, self._b = SimpleNamespace(host=host), body
        async def json(self): return self._b
    pages = [["Mine", ["Coffee"]]]
    try:
        call = lambda host, body: asyncio.run(post(Req(host, body)))
        assert call("192.168.1.20", {"pages": pages, "pin": "0000" if app.state.pin != "0000" else "1111"}).status_code == 403
        assert asyncio.run(pin(Req("192.168.1.20", {}))).status_code == 403          # PIN is only shown on the laptop
        assert call("192.168.1.20", {"pages": pages, "pin": app.state.pin}).status_code == 200
        assert call("127.0.0.1", {"pages": [["Back", ["Tea"]]]}).status_code == 200    # the laptop itself needs none
        assert json.loads(asyncio.run(pin(Req("127.0.0.1", {}))).body)["pin"] == app.state.pin
    finally:
        ph.set_pages(ph.DEFAULT_PAGES, save=False)
