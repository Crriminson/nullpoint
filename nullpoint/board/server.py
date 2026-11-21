"""Nullpoint board server.

    python -m nullpoint.board.server              # live: webcam + your trained models + profile
    python -m nullpoint.board.server --sim        # keyboard only (arrows = glance, space = blink, E = emergency)
    python -m nullpoint.board.server --no-adapt   # live, nystagmus adaptation off (for comparison)
    python -m nullpoint.board.server --lan        # also reachable from a phone on the same Wi-Fi (caregiver view)
    python -m nullpoint.board.server --no-setup   # skip the 10-second setup screen at start (K opens it later)

Open http://127.0.0.1:8765 (board) and http://<laptop-ip>:8765/caregiver (caregiver).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import socket
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from nullpoint.config import TRAINED, PROFILES, DATA, REPORTS, null_screen_x, zone_targets
from nullpoint.board.selector import Selector, SelectorConfig
from nullpoint.board import phrases as ph
from nullpoint.board.tts import Speaker

STATIC = Path(__file__).parent / "static"


class SimGlance:
    """Stand-in models for --sim (no camera): keyboard drives the selector directly."""
    class _H:
        def step(self, p): return p
    hmm = _H()
    def proba(self, g): return [[0, 1, 0]]


class SimBlink:
    def step(self, t, b): return None, {}


def load_profile(path: Path | None):
    p = path or (PROFILES / "profile.json")
    if p.exists():
        d = json.loads(p.read_text())
        return d
    return {}


def build_app(args):
    app = FastAPI(title="Nullpoint")
    speaker = Speaker(enabled=not args.mute)
    profile = load_profile(args.profile)
    freq = profile.get("freq_hz") if profile.get("n_periodic_holds", 0) else None
    null_deg = profile.get("null_deg")
    null_x = 0.5 if args.no_adapt else null_screen_x(profile)
    lock = threading.Lock()
    clients: set[WebSocket] = set()
    meta = {"mode": "sim" if args.sim else "live", "adapt": not args.no_adapt, "freq_hz": freq,
            "null_deg": null_deg, "null_x": null_x, "fps": 0.0}
    app.state.pin = f"{secrets.randbelow(10000):04d}"     # caregiver PIN for editing phrases over Wi-Fi
    guard = {"fails": 0, "until": 0.0}
    onboard = {"null_offered": False}

    if args.sim:
        sel = Selector(SimGlance(), SimBlink(), SelectorConfig(adapt=False), speak=speaker.say)
        cam = None
    else:
        from nullpoint.models.glance import GlanceModel, FEATURE_SETS
        from nullpoint.models.blink import BlinkModel, rule_model
        from nullpoint.capture import Camera
        from nullpoint.features import gaze_features, blink_features
        if (TRAINED / "glance.pkl").exists():
            gm = GlanceModel.load(TRAINED / "glance.pkl")
        else:                                       # new user: the 10-second setup will fit it
            gm = GlanceModel("lda", cols=FEATURE_SETS["eye"])
            print("no glance model yet: the 10-second setup will learn one")
        bm = BlinkModel.load(TRAINED / "blink.pkl") if (TRAINED / "blink.pkl").exists() else rule_model()

        def save_model(info=None):
            if gm.adaptable and (info is None or info.get("ok")):
                TRAINED.mkdir(parents=True, exist_ok=True)
                gm.save(TRAINED / "glance.pkl")     # keeps the model/alignment for next time
                if not (TRAINED / "blink.pkl").exists():
                    bm.save(TRAINED / "blink.pkl")
                if info:
                    print("setup:", info)
            if info and info.get("ok") and meta.get("null_deg") is None and not onboard["null_offered"]:
                onboard["null_offered"] = True             # first setup done: offer the null-zone finder once
                sel.offer_nullfind()

        def save_nullzone(res):
            """Store the webcam estimate; it sets the layout unless a slow-mo null point exists."""
            if not res.get("ok"):
                return ""
            path = args.profile or (PROFILES / "profile.json")
            prof = load_profile(path)
            slowmo = prof.get("null_deg") is not None and prof.get("null_source") != "webcam"
            prof["null_webcam"] = res | {"when": time.strftime("%Y-%m-%d %H:%M")}
            if not slowmo:
                prof["null_deg"] = res["null_deg"] if res["clear"] else 0.0
                prof["null_source"] = "webcam"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prof, indent=1))
            if slowmo:
                return "Slow-motion null point kept; webcam estimate saved for comparison"
            if args.no_adapt:
                return "Saved (adapt is off, layout unchanged)"
            new_x = null_screen_x(prof)
            moved = abs(new_x - meta["null_x"]) > 0.02
            meta.update(null_x=new_x, null_deg=prof["null_deg"])
            sel.cfg.setup_x = zone_targets(new_x)
            if moved:
                sel.offer_setup()
                return "The card moved there; do the 10-second setup for the new layout"
            return ""

        def save_task(summary, trials):
            """Phrase test: full log (per phrase) stays in data/; only the summary goes to reports/."""
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (DATA / "tasks").mkdir(parents=True, exist_ok=True); REPORTS.mkdir(exist_ok=True)
            (DATA / "tasks" / f"task-{stamp}.json").write_text(json.dumps({"summary": summary, "trials": trials}, indent=1))
            out = REPORTS / f"task-{stamp}.json"
            out.write_text(json.dumps(summary, indent=1))
            print("phrase test:", json.dumps(summary))
            return f"reports/{out.name}"

        sel = Selector(gm, bm,
                       # live tuning (user testing: a bit slow, occasional unintended steps); the
                       # benchmarks keep the SelectorConfig defaults so their numbers stay reproducible
                       SelectorConfig(adapt=not args.no_adapt, freq_hz=freq, setup_x=zone_targets(null_x),
                                      repeat_s=0.8, belief_on=0.85),
                       speak=speaker.say, on_setup=save_model, on_nullzone=save_nullzone, on_task=save_task)
        if not gm.adaptable or not args.no_setup:      # a new user can't skip it
            sel.offer_setup()
        video = None
        if args.video:                              # replay a recorded clip instead of the camera
            import glob
            hits = sorted(glob.glob(args.video))
            if not hits:
                raise SystemExit(f"no video matches {args.video}")
            video = hits[-1]
            print("replaying", video)
        cam = Camera(args.cam, source=video)

        def on_frame(ff):
            g, b = gaze_features(ff), blink_features(ff)
            with lock:
                sel.update(ff.t, g, b)
        cam.on_frame(on_frame)

    t0 = time.perf_counter()
    meta["adaptable"] = sel.can_setup

    def now():   # same clock as the camera frames (the selector compares the two)
        return time.perf_counter() - (cam.t0 if cam else t0)

    @app.on_event("startup")
    async def _start():
        if cam:
            cam.start()
        asyncio.create_task(broadcaster())

    @app.on_event("shutdown")
    async def _stop():
        if cam:
            cam.stop()
            with lock:
                save_model()                          # keep self-corrections from this session
        speaker.close()

    async def broadcaster():
        while True:
            await asyncio.sleep(1 / 15)
            if not clients:
                continue
            with lock:
                st = sel.state.as_dict()
                meta["adapt"], meta["freq_hz"] = sel.cfg.adapt, sel._freq()
            meta["fps"] = round(cam.fps, 1) if cam else 0.0
            msg = json.dumps({"state": st, "meta": meta})
            for ws in list(clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    clients.discard(ws)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/game")
    async def game():
        return FileResponse(STATIC / "game.html")

    @app.get("/caregiver")
    async def caregiver():
        return FileResponse(STATIC / "caregiver.html")

    @app.get("/preview.jpg")
    async def preview(request: Request):
        if request.client is None or request.client.host not in ("127.0.0.1", "::1", "localhost"):
            return Response(status_code=403)     # the camera image never leaves the laptop, even with --lan
        jp = cam.latest_jpeg() if cam else None
        return Response(jp or b"", media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def is_local(request: Request) -> bool:
        return request.client is not None and request.client.host in ("127.0.0.1", "::1", "localhost")

    @app.get("/pin")
    async def get_pin(request: Request):
        if not is_local(request):
            return Response(status_code=403)          # shown on the laptop only
        url = f"http://{lan_ip()}:{args.port}/caregiver" if args.lan else None
        return JSONResponse({"pin": app.state.pin, "caregiver_url": url})

    @app.get("/phrases")
    async def get_phrases():
        return JSONResponse({"pages": ph.editable(), "max_pages": ph.MAX_PAGES, "max_per_page": ph.MAX_PER_PAGE,
                             "max_len": ph.MAX_LEN})

    @app.post("/phrases")
    async def post_phrases(request: Request):
        body = await request.json()
        if not is_local(request):
            if time.time() < guard["until"]:
                return JSONResponse({"ok": False, "error": "too many wrong PINs; wait a minute"}, status_code=429)
            if str(body.get("pin", "")) != app.state.pin:
                guard["fails"] += 1
                if guard["fails"] >= 5:
                    guard.update(fails=0, until=time.time() + 60)
                return JSONResponse({"ok": False, "error": "wrong PIN (it is shown in the laptop's help screen, H)"},
                                    status_code=403)
            guard["fails"] = 0
        with lock:
            if sel.task is not None and sel.task.active:
                return JSONResponse({"ok": False, "error": "a phrase test is running"}, status_code=409)
            try:
                ph.reset_defaults() if body.get("reset") else ph.set_pages(body.get("pages"))
            except ValueError as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
            st = sel.state
            st.page = st.page if st.page < len(ph.PAGES) else 0
            st.cursor = min(st.cursor, len(st.tiles()) - 1)
        return JSONResponse({"ok": True, "pages": ph.editable()})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept(); clients.add(ws)
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                cmd = msg.get("cmd")
                with lock:
                    t = now()
                    setup = (sel.state.setup or {}).get("phase")
                    nullf = (sel.state.nullfind or {}).get("phase")
                    task = sel.task
                    if cmd == "setup":
                        sel.offer_setup()
                    elif cmd == "nullzone":
                        sel.offer_nullfind()
                    elif cmd == "task":
                        sel.offer_task()
                    elif cmd in ("quiet_on", "quiet_off"):
                        sel.quiet = cmd == "quiet_on"
                    elif cmd == "sim":
                        sel.cycle_sim()
                    elif cmd == "adapt":
                        if not (task and task.active):        # the test controls Adapt itself
                            sel.set_adapt(not sel.cfg.adapt)
                    elif cmd == "setup_skip":
                        sel.cancel_setup(); sel.cancel_nullfind(); sel.cancel_task()
                    elif (setup == "run" or nullf in ("eyes", "head", "read")) and cmd in ("left", "right", "blink"):
                        pass                                  # setup / null finder running: keys don't act
                    elif cmd == "blink" and nullf == "ready":
                        sel.start_nullfind(t)
                    elif cmd == "blink" and task and task.waiting:
                        task.start(t, sel)
                    elif cmd == "blink" and task and task.finished:
                        sel.cancel_task()
                    elif cmd == "blink" and setup == "ready":
                        sel.start_setup(t)
                    elif cmd == "left":
                        sel._step(-1, t, source="key")
                    elif cmd == "right":
                        sel._step(1, t, source="key")
                    elif cmd == "blink":
                        sel.select(t, source="key")
                    elif cmd == "emergency":
                        sel.emergency(t)
                    elif cmd == "clear":
                        sel.state.emergency = False
                    elif cmd == "speak_again" and sel.state.last_spoken:
                        speaker.say(sel.state.last_spoken)
        except WebSocketDisconnect:
            clients.discard(ws)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main(open_browser=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-adapt", action="store_true")
    ap.add_argument("--lan", action="store_true")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--no-setup", action="store_true", help="don't open the 10-second setup screen at start")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", help="replay a recorded clip (path or glob, newest match) instead of the camera")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--no-browser", action="store_true", help="with python -m nullpoint: don't open a browser tab")
    args = ap.parse_args()
    app = build_app(args)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    print(f"\n  Board:     http://127.0.0.1:{args.port}")
    if args.lan:
        print(f"  Caregiver: http://{lan_ip()}:{args.port}/caregiver  (same Wi-Fi)")
        print(f"  Caregiver PIN for editing phrases: {app.state.pin}  (also in the board's help screen, H)")
    if open_browser and not args.no_browser:
        import webbrowser
        threading.Timer(1.5, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",)).start()
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
