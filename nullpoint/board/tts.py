"""Offline text-to-speech (Windows SAPI5 via pyttsx3) on its own thread so speaking never
blocks the vision loop."""
from __future__ import annotations

import queue
import threading


class Speaker:
    def __init__(self, rate: int = 165, enabled: bool = True):
        self.q: queue.Queue[str | None] = queue.Queue()
        self.enabled = enabled
        self.rate = rate
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def say(self, text: str):
        if self.enabled:
            self.q.put(text)

    def _run(self):
        try:
            import pyttsx3
            import pythoncom   # COM must be initialised on this thread for SAPI5
            pythoncom.CoInitialize()
        except Exception:
            pyttsx3 = None
        eng = None
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                if pyttsx3 is None:
                    print("[tts]", text); continue
                if eng is None:
                    eng = pyttsx3.init(); eng.setProperty("rate", self.rate)
                eng.say(text); eng.runAndWait()
            except Exception as e:
                print("[tts error]", e, "-", text)
                eng = None

    def close(self):
        self.q.put(None)
