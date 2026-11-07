"""Rose v3 server: one process that wires the pipeline and serves the dashboard.

    .venv/bin/python -m rose.app [--port 8091] [--voice zoey] [--local-audio]
                                 [--no-browser-audio] [--ui-only] [--no-keepalive]
                                 [--phone real] [--remote-port 8443 | --no-remote]

FastAPI on 127.0.0.1 carries the dashboard, the browser's audio socket and
the event stream; the MLX runtime thread runs every model; the conductor
thread decides who speaks. A second listener in the same event loop (TLS on
0.0.0.0:8443, rose/remote.py) lets a paired phone reach the same websockets
over Tailscale; it serves nothing else (see REMOTE.md). The server comes up first and the models load
behind it, reporting progress as ``status`` events, so the page is never dead.

Spoken input without a microphone (``POST /api/say``) goes through macOS
``say`` and enters the listener exactly like mic audio, paced in real time,
so typed demos exercise the same VAD, endpointing and barge-in paths a voice
does.

Why a GPU keepalive: macOS drops an idle Metal queue to a low-power state
after about a second, and the next turn pays seconds for the ramp-up. A tiny
op after every 250 ms without GPU work keeps the queue warm for almost
nothing.

``--ui-only`` loads no models: a FakeConductor replays a scripted session
(turns, a barge-in, a backchannel, background phone tasks that really drive
the fake phone, announcements, metrics) on a loop, so the dashboard can be
built and checked without the GPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import asyncio
import contextlib
import json
import math
import queue
import random
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, Response

from .bus import Bus
from .local_audio import MIC_FRAME, MIC_RATE, LocalTransport, SilentTransport
from .phone import Phone
from .realphone import RealPhone
from .remote import (DEFAULT_REMOTE_PORT, REMOTE_CONCURRENCY, LoopbackOnly, Remote, add_pairing_api,
                     bind_socket, build_remote_app, remote_http_protocol)

WEB = Path(__file__).parent / "web"
FRAME_S = MIC_FRAME / MIC_RATE
SAY_LEAD_S = 0.3          # silence before an injected clip (VAD pre-roll)
SAY_TAIL_MAX_S = 3.0      # silence after it while waiting for the listener to commit
KEEPALIVE_IDLE_S = 0.25
REPLACED, AUDIO_OFF = 4001, 4003  # /ws/audio close codes; the dashboard stops reconnecting
UI_ONLY_VOICES = ["zoey", "harper", "everly", "juniper", "easton", "miles"]
NO_STORE = {"Cache-Control": "no-store"}
REPLY_KINDS = frozenset({"user_turn", "rose_partial", "rose_say", "rose_audio_start", "rose_heard",
                         "rose_turn", "tool_call"})


# ------------------------------------------------------------------ audio in
class _Pacer:
    """Sleeps so that successive ticks are ``period`` apart on average."""

    def __init__(self, period: float):
        self.period = period
        self.next = time.monotonic()

    def wait(self) -> None:
        self.next += self.period
        delay = self.next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.25:
            self.next = time.monotonic()  # stalled: do not burst to catch up


class MicFeed:
    """Microphone audio into the listener; ``/api/say`` clips stand in for the mic.

    While a clip plays, live mic frames are dropped: two interleaved streams
    look like noise to the VAD. The clip is framed by silence (before it for
    the VAD pre-roll, after it until the listener commits the turn) because
    endpointing counts audio samples, not wall time, so without a live mic a
    turn would never end.
    """

    def __init__(self, bus: Bus):
        self.bus = bus
        self.sink: Callable[[np.ndarray], None] | None = None
        self.injecting = False
        self.mic_samples = 0
        self.dropped_samples = 0
        self.injected_s = 0.0
        self._clips: queue.Queue[np.ndarray] = queue.Queue()
        self._committed = threading.Event()
        self._level_at = 0.0
        self._level = 0.0
        bus.subscribe(self._on_event)
        threading.Thread(target=self._run, name="say-injector", daemon=True).start()

    def mic(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.mic_samples += samples.size
        self._meter(samples)
        sink = self.sink
        if self.injecting or sink is None:
            self.dropped_samples += samples.size
            return
        sink(samples)

    def inject(self, clip: np.ndarray) -> float:
        clip = np.asarray(clip, dtype=np.float32).reshape(-1)
        self._clips.put(clip)
        return clip.size / MIC_RATE

    def _meter(self, samples: np.ndarray) -> None:
        # Transient 'level' events give non-browser mics a meter on the dashboard.
        if samples.size:
            self._level = max(self._level, float(np.sqrt(np.mean(samples * samples))))
        now = time.monotonic()
        if now - self._level_at >= 0.1:
            self.bus.emit("level", rms=round(self._level, 4))
            self._level_at, self._level = now, 0.0

    def _on_event(self, msg: dict) -> None:
        if msg["kind"] in ("turn_end", "noise"):
            self._committed.set()

    def _push(self, frame: np.ndarray) -> None:
        sink = self.sink
        if sink is not None:
            sink(frame)

    def _run(self) -> None:
        silence = np.zeros(MIC_FRAME, np.float32)
        while True:
            clip = self._clips.get()
            self.injecting = True
            try:
                audio = np.concatenate([np.zeros(int(SAY_LEAD_S * MIC_RATE), np.float32), clip])
                audio = np.pad(audio, (0, -audio.size % MIC_FRAME))
                pacer = _Pacer(FRAME_S)
                for i in range(0, audio.size, MIC_FRAME):
                    self._push(audio[i:i + MIC_FRAME])
                    pacer.wait()
                self._committed.clear()
                tail_end = time.monotonic() + SAY_TAIL_MAX_S
                while time.monotonic() < tail_end and not self._committed.is_set():
                    self._push(silence)
                    pacer.wait()
                self.injected_s += clip.size / MIC_RATE
            except Exception:
                traceback.print_exc()
            finally:
                self.injecting = False


def synthesize_say(text: str, voice: str | None = None) -> np.ndarray:
    """macOS ``say`` -> float32 16 kHz mono (afconvert does the resampling)."""

    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="rose-say-") as tmp:
        src, aiff, wav = (os.path.join(tmp, n) for n in ("text.txt", "say.aiff", "say.wav"))
        Path(src).write_text(text)
        cmd = ["say", "-f", src, "-o", aiff] + (["-v", voice] if voice else [])
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{MIC_RATE}", "-c", "1", aiff, wav],
                       check=True, capture_output=True, timeout=30)
        audio, rate = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != MIC_RATE:
        raise RuntimeError(f"afconvert produced {rate} Hz")
    return audio


# ---------------------------------------------------------------- audio out
class AutoTransport:
    """The browser when one is attached, otherwise a silent real-time client.

    With no browser (Speak box before the mic is started, curl) Rose's audio
    goes to a SilentTransport, so her turns last as long as the speech would
    and a typed barge-in can still land mid-sentence.
    """

    name = "browser"

    def __init__(self):
        from .transport import BrowserTransport

        self.browser = BrowserTransport()
        self.silent = SilentTransport()
        self._on_mic: Callable[[np.ndarray], None] | None = None
        self._on_client: Callable[[dict], None] | None = None

    @property
    def on_mic(self) -> Callable[[np.ndarray], None] | None:
        return self._on_mic

    @on_mic.setter
    def on_mic(self, fn: Callable[[np.ndarray], None] | None) -> None:
        self._on_mic = self.browser.on_mic = fn

    @property
    def on_client(self) -> Callable[[dict], None] | None:
        return self._on_client

    @on_client.setter
    def on_client(self, fn: Callable[[dict], None] | None) -> None:
        self._on_client = self.browser.on_client = fn
        self.silent.on_client = fn or (lambda msg: None)

    @property
    def connected(self) -> bool:
        return self.browser.connected

    def send_audio(self, seg: int, pcm: np.ndarray) -> None:
        (self.browser if self.browser.connected else self.silent).send_audio(seg, pcm)

    def send_control(self, msg: dict) -> None:
        if self.browser.connected:
            self.browser.send_control(msg)
        if self.silent.busy or not self.browser.connected:
            self.silent.send_control(msg)

    async def serve(self, websocket: WebSocket) -> None:
        await self.browser.serve(websocket)


# ------------------------------------------------------------------ keepalive
class GpuKeepalive:
    """A tiny GPU op whenever the runtime has done no GPU work for 250 ms.

    Idleness comes from the runtime's busy-time counters, not its queue:
    long-lived jobs (talker worker, streaming ASR) stay scheduled while idle
    and only spend microseconds per poll.
    """

    def __init__(self, runtime, idle_s: float = KEEPALIVE_IDLE_S, poll_s: float = 0.05,
                 busy_ms: float = 0.5):
        self.runtime, self.idle_s, self.poll_s, self.busy_ms = runtime, idle_s, poll_s, busy_ms
        self.pings = 0
        self._own_s = 0.0
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="gpu-keepalive", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _ping() -> None:
        import mlx.core as mx

        mx.eval(mx.sum(mx.ones((64,), dtype=mx.float32)))

    def _busy_s(self, last: float) -> float:
        try:
            return sum(list(self.runtime.busy_s.values())) - self._own_s
        except RuntimeError:  # counter grew a key mid-read
            return last

    def _run(self) -> None:
        from .runtime import BACKGROUND

        total = self._busy_s(0.0)
        last_work = last_ping = time.monotonic()
        pending = None
        while not self._stop.wait(self.poll_s):
            now = time.monotonic()
            if pending is not None and pending.future.done():
                self._own_s += pending.busy_s
                pending = None
            new_total = self._busy_s(total)
            if (new_total - total) * 1000 > self.busy_ms:
                last_work = now
            total = new_total
            if pending is None and now - last_work >= self.idle_s and now - last_ping >= self.idle_s:
                pending = self.runtime.call(self._ping, priority=BACKGROUND, name="keepalive")
                last_ping = now
                self.pings += 1


# -------------------------------------------------------------- ui-only fake
@dataclass
class _Reply:
    chunks: list[str]
    tool: tuple[str, dict] | None = None
    task: dict | None = None  # {"kind", "goal", "plan" | "answer", "announce"}


@dataclass
class _Barge:
    at_s: float
    text: str
    backchannel: bool = False
    follow_up: _Reply | None = None


@dataclass
class _FakeTask:
    id: int
    kind: str
    goal: str
    status: str = "running"
    steps: list[dict] = field(default_factory=list)
    result: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        end = self.finished or time.time()
        return {"id": self.id, "kind": self.kind, "goal": self.goal, "status": self.status,
                "steps": [dict(s) for s in self.steps], "result": self.result,
                "started": self.started, "finished": self.finished, "detail": self.detail,
                "ms": round((end - self.started) * 1000)}


class _Reset(Exception):
    """Unwinds the scene in progress when the dashboard asks for a reset."""


class FakeConductor:
    """Replays a scripted session through the bus (``--ui-only``).

    Event shapes match the real listener, conductor, speaker and tasks. Text
    sent to ``/api/say`` and notifications from ``/api/notify`` get canned
    answers between scenes, the way the real conductor waits for a quiet
    moment. A reset cuts the current scene short.
    """

    WPS = 2.6  # speaking rate; the dashboard's playback estimate uses the same

    def __init__(self, bus: Bus, phone: Phone):
        self.bus, self.phone = bus, phone
        self.turn_metrics: list[dict] = []
        self.tasks: dict[int, _FakeTask] = {}
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._rid = self._seg = self._task_ids = 0
        self._status = ""
        self._epoch = 0  # bumped by reset: running fake tasks stop
        self._audio_t = 0.0
        self._stop = threading.Event()
        self._reset_req = threading.Event()
        self._lock = threading.Lock()
        self._rng = random.Random(7)

    # ---------------------------------------------------------- interface
    def start(self) -> None:
        threading.Thread(target=self._run, name="fake-conductor", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def inject_text_turn(self, text: str) -> None:
        self._inbox.put(("say", text))

    def on_phone(self, what: str, info: dict) -> None:
        if what == "notification":
            self._inbox.put(("notification", info))

    def reset(self) -> None:
        self._reset_req.set()

    def running(self) -> list[_FakeTask]:
        return [t for t in self.tasks.values() if t.status == "running"]

    # ------------------------------------------------------------- script
    def _run(self) -> None:
        self._set_status("listening")
        while not self._stop.is_set():
            requested = False
            try:
                self._idle(1.2)
                for scene in (self._scene_text_sam, self._scene_calendar, self._scene_notification,
                              self._scene_flour, self._scene_alarm, self._scene_dnd):
                    scene()
                    self._idle(2.2)
                self._idle(1.5)
            except _Reset:
                requested = True
            if not self._stop.is_set():
                self._do_reset(drop_pending=requested)

    def _scene_text_sam(self) -> None:
        self._turn("Text Sam I'm running ten minutes late")

    def _scene_calendar(self) -> None:
        reply = self._canned("What's on my calendar this afternoon?")
        follow = _Reply(["The review is at four, in Room 3B."])
        self._turn("What's on my calendar this afternoon?", reply,
                   barge=_Barge(3.4, "Wait, what time is the review?", follow_up=follow))

    def _scene_notification(self) -> None:
        self.phone.notify("Messages", "Mom", "Can you pick up milk on the way home?")

    def _scene_flour(self) -> None:
        self._turn("How much flour for seven if four needs one and a half cups?")

    def _scene_alarm(self) -> None:
        reply = _Reply(["Okay, setting an alarm for six thirty tomorrow morning.",
                        "I'll label it so you know what it's for."], *self._canned_task("alarm"))
        self._turn("Set an alarm for 6:30 tomorrow", reply, barge=_Barge(1.6, "mm-hmm", backchannel=True))

    def _scene_dnd(self) -> None:
        self._turn("Turn on do not disturb")

    # ------------------------------------------------------------- canned
    def _canned(self, text: str) -> _Reply:
        low = text.lower()
        if "text" in low and "sam" in low:
            return _Reply(["Sure, texting Sam now."], *self._canned_task("sam", text))
        if "calendar" in low or "afternoon" in low:
            return _Reply(["This afternoon you've got the design review with Alex at four, in Room 3B,",
                           "and then dinner with Sam at six thirty at Nopa.",
                           "Want me to remind you before the review?"])
        if "alarm" in low:
            return _Reply(["Okay, setting an alarm for six thirty tomorrow morning."],
                          *self._canned_task("alarm"))
        if "disturb" in low:
            return _Reply(["Turning on Do Not Disturb."], *self._canned_task("dnd"))
        if "flour" in low:
            return _Reply(["Let me work that out."], *self._canned_task("flour"))
        return _Reply(["This is the scripted UI-only mode, so I can't really answer that.",
                       "Restart the server without the UI-only flag to talk to me for real."])

    def _canned_task(self, which: str, text: str = "") -> tuple[tuple[str, dict], dict]:
        if which == "sam":
            msg = text.split("Sam", 1)[-1].strip() or "I'm running ten minutes late"
            msg = msg[0].upper() + msg[1:]
            goal = f"Text Sam Lee: {msg}"
            plan = [[("open_app", "Messages")], [("tap", "Sam Lee")],
                    [("type_text", "compose", msg), ("tap", "send")]]
            task = {"kind": "phone", "goal": goal, "plan": plan, "result": f"Sent Sam Lee \"{msg}\".",
                    "announce": ["Done, Sam has your message."]}
            return ("phone_task", {"goal": goal}), task
        if which == "alarm":
            goal = "Set an alarm for 6:30 AM tomorrow"
            plan = [[("open_app", "Clock")], [("tap", "add_alarm")],
                    [("type_text", "alarm_time", "6:30 AM"), ("type_text", "alarm_label", "Tomorrow"),
                     ("tap", "save")]]
            task = {"kind": "phone", "goal": goal, "plan": plan, "result": "Alarm set for 6:30 AM.",
                    "announce": ["Your alarm is set for six thirty."]}
            return ("phone_task", {"goal": goal}), task
        if which == "dnd":
            goal = "Turn on Do Not Disturb"
            plan = [[("open_app", "Settings")], [("tap", "Do Not Disturb")]]
            task = {"kind": "phone", "goal": goal, "plan": plan, "result": "Do Not Disturb is on.",
                    "announce": ["Do Not Disturb is on now."]}
            return ("phone_task", {"goal": goal}), task
        question = "Flour for 7 servings if 4 servings need 1.5 cups"
        task = {"kind": "think", "goal": question,
                "result": "Two and five-eighths cups, a little over two and a half.",
                "announce": ["For seven you need about two and five-eighths cups,",
                             "so a little over two and a half."]}
        return ("think", {"question": question}), task

    # ------------------------------------------------------------ helpers
    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + max(0.0, seconds)
        while not self._stop.is_set():
            left = end - time.monotonic()
            if left <= 0:
                return
            if self._reset_req.wait(min(left, 0.05)):
                raise _Reset

    def _pause(self, seconds: float) -> None:
        """Sleep for task threads: they stop through the epoch, never by unwinding."""

        self._stop.wait(max(0.0, seconds))

    def _idle(self, seconds: float) -> None:
        """Wait, answering typed turns, notifications and finished tasks meanwhile."""

        end = time.monotonic() + seconds
        while not self._stop.is_set():
            if self._reset_req.is_set():
                raise _Reset
            left = end - time.monotonic()
            if left <= 0:
                return
            try:
                kind, payload = self._inbox.get(timeout=min(left, 0.05))
            except queue.Empty:
                continue
            if kind == "say":
                self._turn(str(payload))
            elif kind == "notification":
                first = str(payload.get("title", "Someone")).split()[0]
                self._announce([f"{first} just texted: {payload.get('text', '')}"])
            elif kind == "task_done":
                self._announce(payload)

    def _do_reset(self, drop_pending: bool) -> None:
        with self._lock:
            self._epoch += 1
            self.tasks.clear()
        while drop_pending:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break
        self._reset_req.clear()
        self.phone.reset()
        self.bus.emit("reset")
        self._status = ""
        self._set_status("listening")

    def _set_status(self, state: str) -> None:
        if state != self._status:
            self._status = state
            self.bus.emit("status", state=state)

    def _vad(self, p: float, speaking: bool) -> None:
        self.bus.emit("vad", p=round(p, 3), speaking=speaking)

    def _next_seg(self) -> int:
        self._seg += 1
        return self._seg

    def _jitter(self, lo: float, hi: float) -> float:
        return round(self._rng.uniform(lo, hi))

    @classmethod
    def _say_s(cls, text: str) -> float:
        return max(0.6, len(text.split()) / cls.WPS)

    # -------------------------------------------------------------- turns
    def _turn(self, text: str, reply: _Reply | None = None, barge: _Barge | None = None) -> None:
        rid, listen = self._user_speaks(text)
        self._rose_reply(rid, reply or self._canned(text), listen, barge)

    def _user_speaks(self, text: str, said: int = 0) -> tuple[int, dict]:
        """The user says ``text``; ``said`` words were already heard (a barge-in)."""

        words = text.split()
        if not said:
            self._set_status("user_speaking")
            self.bus.emit("speech_start", t_audio=round(self._audio_t, 3))
        for i in range(said, len(words)):
            self._vad(0.86 + 0.1 * self._rng.random(), True)
            self._sleep(0.27)
            self._audio_t += 0.27
            self.bus.emit("partial", text=" ".join(words[: i + 1]))
        self.bus.emit("speech_pause", t_audio=round(self._audio_t, 3))
        self._vad(0.04, True)
        listen = {"endpoint_ms": self._jitter(190, 270), "asr_ms": self._jitter(28, 46),
                  "smart_turn_p": round(self._rng.uniform(0.86, 0.98), 3)}
        self._sleep(listen["endpoint_ms"] / 1000)
        self._audio_t += listen["endpoint_ms"] / 1000
        self.bus.emit("turn_end", text=text, t_audio=round(self._audio_t, 3), **listen)
        self._vad(0.02, False)
        self._rid += 1
        self.bus.emit("user_turn", text=text, id=self._rid)
        self._set_status("thinking")
        return self._rid, listen

    def _rose_reply(self, rid: int, reply: _Reply, listen: dict | None, barge: _Barge | None = None,
                    announcement: bool = False) -> None:
        self._set_status("thinking")
        llm_ms = self._jitter(70, 120)
        self._sleep(llm_ms / 1000)
        first = reply.chunks[0]
        words = first.split()
        for k in range(1, len(words) + 1):  # streamed at ~85 tok/s
            self.bus.emit("rose_partial", id=rid, text=" ".join(words[:k]))
            self._sleep(0.012)
        text_ms = len(words) * 12
        tts_ms = self._jitter(18, 40)
        self.bus.emit("rose_say", id=rid, seg=self._next_seg(), text=first, spoken=first)
        self._sleep(tts_ms / 1000)
        endpoint = (listen or {}).get("endpoint_ms", 0)
        v2v = endpoint + llm_ms + text_ms + tts_ms
        self.bus.emit("rose_audio_start", id=rid, latency_ms=None if announcement else v2v, tts_ms=tts_ms)
        self._set_status("speaking")
        metrics = {"id": rid, "announcement": announcement}
        if not announcement:
            metrics.update({"voice_to_voice_ms": v2v, "llm_first_token_ms": endpoint + llm_ms,
                            "first_speech_text_ms": endpoint + llm_ms + 4, **(listen or {})})
        self.turn_metrics.append(metrics)
        self.bus.emit("metrics", turn=metrics)

        spoken = first
        for chunk in reply.chunks[1:]:
            spoken = f"{spoken} {chunk}"
            self.bus.emit("rose_partial", id=rid, text=spoken)
            self.bus.emit("rose_say", id=rid, seg=self._next_seg(), text=chunk, spoken=chunk)
        if reply.tool is not None:
            name, args = reply.tool
            self._sleep(0.15)
            self.bus.emit("tool_call", id=rid, name=name, arguments=args)
            if reply.task is not None:
                self._start_task(reply.task)

        total = sum(self._say_s(c) for c in reply.chunks)
        if barge is None or barge.at_s >= total:
            self._sleep(total)
            self._finish(rid, spoken, interrupted=False)
            return
        self._sleep(barge.at_s)
        self.bus.emit("barge", stage="duck", reason="user started speaking")
        self._set_status("user_speaking")
        self.bus.emit("speech_start", t_audio=round(self._audio_t, 3))
        if barge.backchannel:
            self._vad(0.74, True)
            self._sleep(0.32)
            self.bus.emit("partial", text=barge.text)
            self.bus.emit("speech_pause", t_audio=round(self._audio_t, 3))
            self._vad(0.05, False)
            self._sleep(0.28)
            self.bus.emit("barge", stage="resume", reason=f"backchannel: {barge.text}")
            self._set_status("speaking")
            # The listener still ends the utterance; the conductor makes no user turn of it.
            self.bus.emit("turn_end", text=barge.text, t_audio=round(self._audio_t, 3), endpoint_ms=280,
                          asr_ms=self._jitter(28, 46), smart_turn_p=0.9)
            self._sleep(max(0.0, total - barge.at_s - 0.6))
            self._finish(rid, spoken, interrupted=False)
            return
        words = barge.text.split()
        self._vad(0.91, True)
        self._sleep(0.2)
        self.bus.emit("partial", text=" ".join(words[:2]))
        self._sleep(0.18)
        self.bus.emit("barge", stage="stop", reason="user is talking", ms=380)
        heard_words = int((barge.at_s + 0.38) * self.WPS)
        heard = " ".join(spoken.split()[:heard_words])
        self._finish(rid, heard, interrupted=True, status="user_speaking")
        rid2, listen = self._user_speaks(barge.text, said=2)
        self._rose_reply(rid2, barge.follow_up or self._canned(barge.text), listen)

    def _finish(self, rid: int, text: str, interrupted: bool, status: str = "listening") -> None:
        self.bus.emit("rose_heard", id=rid, text=text, interrupted=interrupted)
        if text.strip():
            self.bus.emit("rose_turn", id=rid, text=text, interrupted=interrupted)
        self._set_status(status)

    def _announce(self, chunks: list[str]) -> None:
        self._rid += 1
        self._rose_reply(self._rid, _Reply(list(chunks)), None, announcement=True)

    # -------------------------------------------------------------- tasks
    def _start_task(self, spec: dict) -> None:
        with self._lock:
            self._task_ids += 1
            task = _FakeTask(self._task_ids, spec["kind"], spec["goal"])
            self.tasks[task.id] = task
            epoch = self._epoch
        self.bus.emit("task", task=task.to_dict())
        threading.Thread(target=self._task_body, args=(task, spec, epoch),
                         name=f"fake-task{task.id}", daemon=True).start()

    def _task_body(self, task: _FakeTask, spec: dict, epoch: int) -> None:
        try:
            if task.kind == "think":
                self._think(task, epoch)
            else:
                self._phone_task(task, spec["plan"], epoch)
        except Exception as exc:
            task.status, task.result = "failed", f"{type(exc).__name__}: {exc}"
        if epoch != self._epoch:
            return
        if task.status == "running":
            task.status, task.result = "done", spec["result"]
        task.finished = time.time()
        task.detail = ""
        self.bus.emit("task", task=task.to_dict())
        self._inbox.put(("task_done", spec["announce"]))

    def _think(self, task: _FakeTask, epoch: int) -> None:
        for n in (200, 400):
            self._pause(1.1)
            task.detail = f"thinking ({n} tokens)"
            self.bus.emit("task", task=task.to_dict())
        self._pause(0.6)
        task.steps.append({"n": 1, "ms": self._jitter(2700, 3100), "tokens": 512,
                           "what": "thought for 512 tokens"})
        self.bus.emit("task", task=task.to_dict())
        self._pause(0.4)
        task.steps.append({"n": 2, "ms": self._jitter(280, 360), "tokens": 24, "what": "answered"})

    def _phone_task(self, task: _FakeTask, plan: list[list[tuple]], epoch: int) -> None:
        view = f"task{task.id}"
        session = self.phone.session(view)
        try:
            for n, actions in enumerate(plan, 1):
                task.detail = f"step {n}"
                self.bus.emit("task", task=task.to_dict())
                t0 = time.monotonic()
                self._pause(self._rng.uniform(0.35, 0.55))  # the agent decoding its next call
                done = []
                for act in actions:
                    if epoch != self._epoch or self._stop.is_set():
                        return
                    done.append(self._act(session, view, *act))
                    self._pause(0.25)
                screen = self.phone.ui_tree(view).split("\n", 1)[0].removeprefix("Screen: ")
                task.steps.append({"n": n, "ms": round((time.monotonic() - t0) * 1000),
                                   "tokens": 18 * len(actions), "actions": done,
                                   "what": ", ".join(_what(a) for a in done), "screen": screen})
                self.bus.emit("task", task=task.to_dict())
            task.steps.append({"n": len(plan) + 1, "ms": self._jitter(160, 240), "tokens": 22,
                               "actions": [{"name": "finish", "args": {}, "ok": True}],
                               "what": "finished"})
        finally:
            session.close()

    def _act(self, session, view: str, name: str, target: str = "", text: str = "") -> dict:
        label = None
        if name == "open_app":
            r = session.open_app(target)
            args: dict = {"app": target}
        else:
            el = next((e for e in self.phone.screen(view).elements
                       if e.id == target or e.label == target), None)
            mark = str(el.mark) if el is not None else target
            label = el.label if el is not None else None
            args = {"id": mark} | ({"text": text} if name == "type_text" else {})
            r = session.type_text(mark, text) if name == "type_text" else session.tap(mark)
        out = {"name": name, "args": args, "ok": r.ok, "result": r.message, "changed": r.changed}
        if label:
            out["label"] = label
        return out


def _what(a: dict) -> str:
    name, args = a["name"], a["args"]
    what = {"open_app": f"opened {args.get('app', '')}",
            "tap": f"tapped {a.get('label') or args.get('id', '')}",
            "type_text": f"typed '{args.get('text', '')}'"}.get(name, name)
    return what if a["ok"] else what + " (failed)"


# ------------------------------------------------------------------- server
def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple, frozenset)):
        return list(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def _finite(o: Any) -> Any:
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _owner_ids(msg: dict) -> tuple[int | None, int | None]:
    """(reply id, task id) an event belongs to."""

    kind = msg["kind"]
    if kind in REPLY_KINDS:
        rid = msg.get("id")
        return (rid if isinstance(rid, int) else None), None
    if kind == "task":
        tid = (msg.get("task") or {}).get("id")
        return None, (tid if isinstance(tid, int) else None)
    return None, None


def dumps(msg: Any) -> str:
    try:
        return json.dumps(msg, default=_jsonable, allow_nan=False)
    except ValueError:  # NaN / inf somewhere
        return json.dumps(_finite(json.loads(json.dumps(msg, default=_jsonable))))


class Server:
    """Owns the pipeline objects and everything the HTTP routes need."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mode = "ui-only" if args.ui_only else "live"
        self.bus = Bus()
        self.transcripts = None
        if getattr(args, 'brain', 'local') in ('codex', 'claude'):
            from .transcripts import Transcripts
            self.transcripts = Transcripts(getattr(args, 'transcript_dir', None))
        # --phone real: the user's Android phone through Rose Agent (rose/realphone.py), not the fake one.
        self.phone = RealPhone(bus=self.bus) if getattr(args, "phone", "fake") == "real" else Phone(bus=self.bus)
        self.mic = MicFeed(self.bus)
        self.voice = args.voice
        self.ready = False
        self.state: dict = {"state": "loading", "detail": "Starting"}
        self.load_s: dict[str, float] = {}
        self.runtime = self.brain = self.task_brain = self.listener = self.speaker = self.tasks = None
        self.conductor: Any = None
        self.keepalive: GpuKeepalive | None = None
        self.transport: Any = None
        self.audio_ws: WebSocket | None = None
        self._last_ids = (0, 0)  # highest (reply, task) id seen
        self._floors = (0, 0)    # ids at the last reset: their late events are not shown
        self._ids_lock = threading.Lock()
        self._stopping = threading.Event()
        self.bus.subscribe(self._watch)
        self.phone.listeners.append(self._on_phone)

    # ----------------------------------------------------------- lifecycle
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.bus.loop = loop
        threading.Thread(target=self._stats_loop, name="stats", daemon=True).start()
        if self.mode == "ui-only":
            self.conductor = FakeConductor(self.bus, self.phone)
            self.conductor.start()
            self.ready = True
            self.bus.emit("ready", seconds=0, load_s={})
        else:
            self.bus.emit("status", state="loading", detail="Starting the MLX runtime")
            threading.Thread(target=self._load, name="loader", daemon=True).start()

    def shutdown(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        for obj, method in ((self.listener, "stop"), (self.conductor, "stop"),
                            (self.keepalive, "stop"), (self.transport, "stop"), (self.runtime, "stop")):
            if obj is not None and hasattr(obj, method):
                with contextlib.suppress(Exception):
                    getattr(obj, method)()
        # Close Rose's workspace on the phone (RealPhone.close() waits on the link: on a worker
        # thread, bounded, so a slow or gone phone cannot hold up the exit). The fake phone has none.
        close = getattr(self.phone, "close", None)
        if callable(close):
            def close_quietly() -> None:
                with contextlib.suppress(Exception):
                    close()

            t = threading.Thread(target=close_quietly, name="workspace-close", daemon=True)
            t.start()
            t.join(3.0)

    def _make_transport(self) -> Any:
        a = self.args
        if a.local_audio:
            transport = LocalTransport(_device(a.input_device), _device(a.output_device))
        elif not a.browser_audio:
            transport = SilentTransport()
        else:
            transport = AutoTransport()
        transport.on_mic = self.mic.mic
        transport.on_client = self._on_client
        return transport

    def _step(self, name: str, detail: str, fn: Callable[[], Any]) -> Any:
        self.bus.emit("status", state="loading", detail=detail)
        t = time.monotonic()
        result = fn()
        self.load_s[name] = round(time.monotonic() - t, 2)
        print(f"[rose] {name}: {self.load_s[name]:.1f} s", flush=True)
        return result

    def _load(self) -> None:
        t0 = time.monotonic()
        try:
            codex = getattr(self.args, 'brain', 'local') in ('codex', 'claude')
            cloud = getattr(self.args, 'cloud_audio', None)
            if cloud:
                from .cloud_audio import prepare
                prepare()
            from .brain import REPO as BRAIN_REPO, Brain
            from .conductor import Conductor
            from .listener import Listener
            from .runtime import Runtime
            from .speaker import Speaker
            from .tasks import TaskManager

            self.transport = self._make_transport()
            self.runtime = rt = Runtime()
            if not codex:
                brain_repo = BRAIN_REPO if self.args.brain in (None, "local") else self.args.brain
                task_repo = self.args.task_brain or os.environ.get("ROSE_TASK_BRAIN") or brain_repo
                self.brain = self._step("brain", f"Loading the brain ({brain_repo.split('/')[-1]})",
                                        lambda: rt.run(lambda: Brain(brain_repo), name="brain-load", timeout=900))
                self.task_brain = self.brain
                if task_repo != brain_repo:  # a second model for the phone agent and thinker
                    self.task_brain = self._step(
                        "task-brain", f"Loading the task brain ({task_repo.split('/')[-1]})",
                        lambda: rt.run(lambda: Brain(task_repo), name="task-brain-load", timeout=900))
            self.listener = Listener(rt, self.bus, on_event=self._on_listen)
            if not cloud:
                self._step("listener", "Loading ears (VAD, streaming ASR, Parakeet, smart-turn)",
                           self.listener.load)
            self.speaker = Speaker(rt, self.bus, self.transport)
            self.speaker.on_audible = self.listener.set_rose_audible  # echo guard
            if cloud:
                from .cloud_audio import load
                self._step("voice", "Loading Linux voice providers", lambda: load(cloud, rt, self.listener, self.speaker, self.voice))
            else:
                self._step("speaker", f"Loading the voice ({self.voice})", lambda: self.speaker.load(self.voice))
            if codex:
                from .codex_conductor import CodexConductor
                if self.args.brain == 'claude':
                    from .codex_client import ClaudeClient
                    self.conductor = CodexConductor(self.phone, self.bus, self.speaker, self.transcripts,
                                                    model='claude', client_factory=ClaudeClient)
                else:
                    self.conductor = CodexConductor(self.phone, self.bus, self.speaker, self.transcripts)
                self.tasks = self.conductor
            else:
                self.tasks = TaskManager(rt, self.task_brain, self.phone, self.bus, on_done=self._on_task_done)
                self._step("tasks", "Preparing the phone agent and thinker", self.tasks.prepare)
                self.conductor = Conductor(rt, self.brain, self.listener, self.speaker, self.tasks,
                                           self.phone, self.bus)
            self._step("conductor", "Preparing Rose's conversation", self.conductor.start)
            self.listener.start()
            self.mic.sink = self.listener.push
            if isinstance(self.transport, LocalTransport):
                self.transport.start()
            if self.args.keepalive and not cloud:
                self.keepalive = GpuKeepalive(rt)
                self.keepalive.start()
            self.ready = True
            seconds = round(time.monotonic() - t0, 1)
            print(f"[rose] ready in {seconds} s", flush=True)
            self.bus.emit("ready", seconds=seconds, load_s=dict(self.load_s))
        except Exception as exc:
            if getattr(self.args, 'brain', 'local') not in ('codex', 'claude'):
                traceback.print_exc()
            self.bus.emit("error", message=(f"startup failed: {type(exc).__name__}; check the Claude Code subscription login (claude /login)" if self.args.brain == 'claude' else f"startup failed: {type(exc).__name__}; check Codex ChatGPT login and Terra Fast availability") if codex else f"startup failed: {type(exc).__name__}: {exc}")
            self.bus.emit("status", state="loading",
                          detail=f"Startup failed ({type(exc).__name__}); see the server log")

    # -------------------------------------------------------------- wiring
    def _on_listen(self, ev) -> None:
        if self.conductor is not None:
            self.conductor.on_listen(ev)

    def _on_task_done(self, task) -> None:
        if self.conductor is not None:
            self.conductor.on_task_done(task)

    def _on_phone(self, what: str, info: dict | None = None) -> None:
        if self.conductor is not None:
            self.conductor.on_phone(what, info or {})

    def _on_client(self, msg: dict) -> None:
        if self.speaker is not None:
            self.speaker.on_client(msg)
        hook = getattr(self.conductor, "on_client", None)
        if hook is not None:
            hook(msg)

    def _watch(self, msg: dict) -> None:
        kind = msg["kind"]
        rid, tid = _owner_ids(msg)
        if rid is not None or tid is not None:
            with self._ids_lock:  # emitted from the conductor and runtime threads
                last_r, last_t = self._last_ids
                self._last_ids = (max(last_r, rid or 0), max(last_t, tid or 0))
        if kind == "status":
            self.state = {"state": msg.get("state"), "detail": msg.get("detail", "")}
        elif kind == "reset":
            # A reply or task cut short by the reset still reports afterwards (the
            # stopped reply's rose_heard, the cancelled task): dashboards skip those.
            with self._ids_lock:
                self._floors = self._last_ids
            # A dashboard that connects later starts from the reset, not the old talk.
            hist = self.bus.history
            with contextlib.suppress(Exception):
                hist.clear()
                hist.append(msg)
                hist.append({"kind": "status", "t": msg["t"], **self.state})

    def _stale(self, msg: dict) -> bool:
        rid, tid = _owner_ids(msg)
        reply_floor, task_floor = self._floors
        return (rid is not None and 0 < rid <= reply_floor) or (tid is not None and tid <= task_floor)

    def _stats_loop(self) -> None:
        while not self._stopping.wait(1.0):
            with contextlib.suppress(Exception):
                self.bus.emit("stats", **self.stats())

    # --------------------------------------------------------------- views
    def voices(self) -> list[str]:
        if self.speaker is not None:
            return list(self.speaker.voices)
        return UI_ONLY_VOICES if self.mode == "ui-only" else [self.voice]

    def current_voice(self) -> str:
        return getattr(self.speaker, "voice", None) or self.voice

    def task_list(self) -> list[dict]:
        tasks = self.tasks if self.mode == "live" else self.conductor
        if tasks is None:
            return []
        items = getattr(tasks, "tasks", None)
        items = list(items.values()) if isinstance(items, dict) else tasks.running()
        items = [t for t in items if t.id > self._floors[1]]  # TaskManager keeps pre-reset tasks
        snapshot = getattr(tasks, 'task_snapshot', lambda task: task.to_dict())
        return [snapshot(t) for t in sorted(items, key=lambda t: t.id)[-20:]]

    def audio_stats(self) -> dict:
        t = self.transport
        if t is None:
            output = "none"
        elif isinstance(t, AutoTransport):
            output = "browser" if t.connected else "silent (no browser attached)"
        else:
            output = t.name
        return {"output": output, "browser_client": self.audio_ws is not None,
                "mic_s": round(self.mic.mic_samples / MIC_RATE, 1),
                "dropped_s": round(self.mic.dropped_samples / MIC_RATE, 1),
                "injected_s": round(self.mic.injected_s, 1), "injecting": self.mic.injecting}

    def stats(self) -> dict:
        out: dict[str, Any] = {"mode": self.mode, "ready": self.ready, "audio": self.audio_stats()}
        if self.runtime is not None:
            out["runtime"] = self.runtime.stats()
        if self.listener is not None and hasattr(self.listener, "stats"):
            out["listener"] = self.listener.stats()
        if self.speaker is not None and hasattr(self.speaker, "buffered_s"):
            out["speaker"] = {"buffered_s": round(self.speaker.buffered_s(), 2),
                              "audible": bool(self.speaker.audible)}
        if self.keepalive is not None:
            out["keepalive"] = {"pings": self.keepalive.pings}
        return out

    def snapshot(self) -> dict:
        return {"mode": self.mode, "ready": self.ready, "status": self.state,
                "voice": self.current_voice(), "voices": self.voices(),
                "phone": self.phone.snapshot(), "ui_tree": self.phone.ui_tree(),
                "tasks": self.task_list(), "load_s": self.load_s,
                "metrics": list(getattr(self.conductor, "turn_metrics", []))[-50:],
                "stats": self.stats(),
                "conversation_id": getattr(self.conductor, 'conversation', None),
                "model": getattr(self.conductor, 'model', None)}

    # ------------------------------------------------------------- actions
    async def say(self, text: str, typed: bool = False) -> tuple[dict, int]:
        if self.mode == "ui-only":
            self.conductor.inject_text_turn(text)
            return {"ok": True, "mode": "scripted"}, 200
        if not self.ready:
            return {"ok": False, "error": "Rose is still loading"}, 503
        if typed or getattr(self.args, 'cloud_audio', None):
            self.conductor.inject_text_turn(text)
            return {"ok": True, "mode": "typed"}, 200
        try:
            audio = await asyncio.to_thread(synthesize_say, text)
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            return {"ok": False, "error": f"say failed: {type(exc).__name__}"}, 500
        seconds = self.mic.inject(audio)
        return {"ok": True, "mode": "speech", "seconds": round(seconds, 2)}, 200

    def reset(self, phone: bool = True) -> dict:
        if self.tasks is not None:
            for t in self.tasks.running():
                self.tasks.cancel(t.id)
        if phone:
            self.phone.reset()
        if self.conductor is not None:
            self.conductor.reset()
        return {"ok": True}

    def set_voice(self, voice: str) -> tuple[dict, int]:
        if voice not in self.voices():
            return {"ok": False, "error": f"unknown voice; choose from {', '.join(self.voices())}"}, 400
        if self.speaker is not None:
            try:
                self.speaker.set_voice(voice)  # loads the voice prompt on the runtime thread
            except Exception as exc:
                return {"ok": False, "error": f"voice change failed: {type(exc).__name__}"}, 500
        self.voice = voice
        self.bus.emit("voice", voice=voice)
        return {"ok": True, "voice": voice}, 200

    # ----------------------------------------------------------- websockets
    async def serve_audio(self, ws: WebSocket) -> None:
        await ws.accept()
        browser = self.mode == "ui-only" or isinstance(self.transport, AutoTransport) \
            or (self.transport is None and self.args.browser_audio and not self.args.local_audio)
        if not browser:
            await ws.close(code=AUDIO_OFF, reason="browser audio is off on the server")
            return
        old, self.audio_ws = self.audio_ws, ws
        if old is not None:
            with contextlib.suppress(Exception):
                await old.close(code=REPLACED, reason="another dashboard took over the audio")
        try:
            if isinstance(self.transport, AutoTransport):
                await self.transport.serve(ws)
            else:
                await self._drain_audio(ws)  # ui-only, or models still loading
        finally:
            if self.audio_ws is ws:
                self.audio_ws = None

    async def _drain_audio(self, ws: WebSocket) -> None:
        """Mic frames before a transport exists: meter them, then drop them."""

        with contextlib.suppress(Exception):
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                data = msg.get("bytes")
                if data:
                    self.mic.mic(np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
                                 .astype(np.float32) / 32768.0)
                if isinstance(self.transport, AutoTransport):
                    # Models finished loading while we were connected: hand over.
                    await self.transport.serve(ws)
                    return

    async def serve_phone(self, ws: WebSocket, authenticated: bool = False) -> None:
        """Rose Agent's command link. ``authenticated``: it came through the remote listener's
        signature check (an unauthenticated USB link never replaces such a link)."""

        if not isinstance(self.phone, RealPhone):
            await ws.accept()
            await ws.close(code=4003, reason="Rose is using the fake phone (start with --phone real)")
            return
        await self.phone.link.serve(ws, authenticated=authenticated)

    async def serve_events(self, ws: WebSocket) -> None:
        await ws.accept()
        q = self.bus.subscribe_queue()
        try:
            history = [m for m in list(self.bus.history) if not self._stale(m)]
            seen = {id(m) for m in history}
            state = self.snapshot()
            await ws.send_text(dumps({"kind": "hello", "t": time.time(), "history": len(history),
                                      "state": state}))
            for m in history:
                await ws.send_text(dumps({**m, "replay": True}))
            await ws.send_text(dumps({"kind": "replay_done", "t": time.time()}))
            # Phone clients intentionally skip historical replay. Restore the
            # current task collection as live snapshots, using the same scoped
            # identity/binding as subsequent updates; never replay user actions.
            for task in state.get('tasks', []):
                await ws.send_text(dumps({'kind': 'task', 't': time.time(), 'task': task}))
            incoming = asyncio.create_task(ws.receive())
            outgoing = asyncio.create_task(q.get())
            try:
                while True:
                    ready, _ = await asyncio.wait((incoming, outgoing), return_when=asyncio.FIRST_COMPLETED)
                    if incoming in ready:
                        frame = incoming.result()
                        if frame['type'] == 'websocket.disconnect':
                            break
                        incoming = asyncio.create_task(ws.receive())
                    if outgoing in ready:
                        m = outgoing.result()
                        outgoing = asyncio.create_task(q.get())
                        if id(m) not in seen and not self._stale(m):
                            await ws.send_text(dumps(m))
            finally:
                incoming.cancel()
                outgoing.cancel()
                await asyncio.gather(incoming, outgoing, return_exceptions=True)
        except Exception:
            pass  # client went away
        finally:
            self.bus.unsubscribe_queue(q)


def _device(value: str | None) -> str | int | None:
    if value is None:
        return None
    return int(value) if value.strip().isdigit() else value


def _json(payload: Any, status: int = 200) -> Response:
    return Response(dumps(payload), status_code=status, media_type="application/json",
                    headers=NO_STORE)


def build_app(server: Server, remote: Remote | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        server.start(asyncio.get_running_loop())
        seen = asyncio.create_task(remote.keep_seen()) if remote is not None else None
        yield
        if seen is not None:
            seen.cancel()
        await asyncio.to_thread(server.shutdown)

    app = FastAPI(title="Rose", lifespan=lifespan)
    # No login on the dashboard: loopback Host and Origin only, so a web page cannot use it.
    app.add_middleware(LoopbackOnly, port=server.args.port)
    add_pairing_api(app, remote, server.args.port)  # /api/pair/*: QR invites and paired devices

    def static(name: str, media_type: str) -> Callable[[], FileResponse]:
        def route() -> FileResponse:
            return FileResponse(WEB / name, media_type=media_type, headers=NO_STORE)
        route.__name__ = "static_" + name.replace(".", "_")
        return route

    app.get("/")(static("index.html", "text/html"))
    for name, media in (("audio.js", "text/javascript"), ("app.js", "text/javascript"),
                        ("app.css", "text/css")):
        app.get(f"/{name}")(static(name, media))

    @app.get("/phone.png")
    def phone_png(marks: int = 0) -> Response:
        return Response(server.phone.screenshot_png(marks=bool(marks)), media_type="image/png",
                        headers=NO_STORE)

    @app.get("/api/phone")
    def phone_state() -> Response:
        return _json(server.phone.snapshot())

    @app.get("/api/state")
    def state() -> Response:
        return _json(server.snapshot())

    @app.get("/api/transcripts")
    def transcripts() -> Response:
        return _json(server.transcripts.conversations() if server.transcripts else [])

    @app.get("/api/transcripts/{conversation}")
    def transcript(conversation: str) -> Response:
        if not server.transcripts:
            return _json({"error": "transcripts unavailable"}, 404)
        return _json(server.transcripts.export(conversation))

    @app.post("/api/tasks/{task_id}/stop")
    def stop_task(task_id: int, body: dict) -> Response:
        # A stale browser from another server lifetime cannot stop a reused id.
        if body.get('conversation_id') != getattr(server.conductor, 'conversation', None):
            return _json({"ok": False, "error": "stale conversation"}, 409)
        return _json({"ok": bool(server.tasks and server.tasks.cancel(task_id))})

    @app.post("/api/say")
    async def say(body: dict) -> Response:
        text = " ".join(str(body.get("text", "")).split())[:400]
        if not text:
            return _json({"ok": False, "error": "no text"}, 400)
        payload, status = await server.say(text, typed=bool(body.get("typed")))
        return _json(payload, status)

    @app.post("/api/notify")
    def notify(body: dict) -> Response:
        text = str(body.get("text", "")).strip()[:300]
        if not text:
            return _json({"ok": False, "error": "no text"}, 400)
        server.phone.notify(str(body.get("app") or "Messages"), str(body.get("title") or "Someone"), text)
        return _json({"ok": True})

    @app.post("/api/reset")
    def reset(body: dict | None = None) -> Response:
        return _json(server.reset(phone=bool((body or {}).get("phone", True))))

    @app.post("/api/voice")
    def voice(body: dict) -> Response:
        payload, status = server.set_voice(str(body.get("voice", "")).strip().lower())
        return _json(payload, status)

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket) -> None:
        await server.serve_audio(ws)

    @app.get("/api/realphone")
    def phone_status(look: str = "") -> Response:
        """Real phone link status; look=apps|screen|workspace adds a read-only observation
        (never an action; workspace is the background controller's state)."""

        if not isinstance(server.phone, RealPhone):
            return _json({"real": False})
        link = server.phone.link
        out: dict = {"real": True, "connected": link.connected, "model": link.model,
                     # Rose's workspace as Rose sees it (cached, no phone call): the app open there,
                     # its session, why it last closed, seconds until the idle close.
                     "background": server.phone.background_status()}
        if link.connected:
            out["control"] = link.call("control_status", timeout=8)
            if look == "apps":
                out["apps"] = link.call("list_apps", timeout=8)
            elif look == "screen":
                out["screen"] = link.call("inspect_ui", timeout=8)
            elif look == "workspace":
                out["workspace"] = link.call("app_command", timeout=8, packageName="dev.pixel.backgroundcontrol",
                                             operation="get_state", params={})
        return _json(out)

    @app.websocket("/ws/phone")
    async def ws_phone(ws: WebSocket) -> None:
        # Phone control always uses the signed Tailscale endpoint when remote mode is on.
        if remote is not None:
            await ws.accept()
            await ws.close(code=PAIRED_ONLY, reason="Pair Rose Agent over Tailscale; unsigned USB phone control is unavailable")
            return
        await server.serve_phone(ws)

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await server.serve_events(ws)

    return app


# SIGHUP too: closing the terminal should stop Rose cleanly (main's finally saves lastSeen).
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
PAIRED_ONLY = 4009  # Phone control uses the signed Tailscale listener, never implicit USB.


class _Uvicorn(uvicorn.Server):
    """A uvicorn server that leaves signals to ``serve_all`` (two servers share one loop)."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


def build_servers(server: Server, args: argparse.Namespace,
                  remote: Remote | None) -> list[tuple[uvicorn.Server, list | None]]:
    """The dashboard server and, unless disabled or its port is busy, the remote TLS listener."""

    dash = uvicorn.Config(build_app(server, remote), host="127.0.0.1", port=args.port, log_level=args.log_level)
    servers: list[tuple[uvicorn.Server, list | None]] = [(_Uvicorn(dash), None)]
    if remote is not None:
        try:
            sock = bind_socket(args.remote_port)
        except OSError as exc:
            print(f"[rose] remote listener off: port {args.remote_port} unavailable ({exc.strerror}); "
                  "phones can still use USB", flush=True)
        else:
            cfg = uvicorn.Config(build_remote_app(server, remote), host="0.0.0.0", port=args.remote_port,
                                 ssl_keyfile=str(remote.identity.key_path),
                                 ssl_certfile=str(remote.identity.cert_path),
                                 lifespan="off", proxy_headers=False, server_header=False,
                                 access_log=False, log_level=args.log_level,
                                 http=remote_http_protocol(), limit_concurrency=REMOTE_CONCURRENCY)
            servers.append((_Uvicorn(cfg), [sock]))
            remote.listening = True
    return servers


async def serve_all(servers: list[tuple[uvicorn.Server, list | None]],
                    shutdown: Callable[[], None] | None = None) -> None:
    """Run the servers on this loop until a signal (or one of them stops), then stop them all."""

    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()

    def on_signal() -> None:
        if stopping.is_set():
            for s, _ in servers:
                s.force_exit = True
                s.should_exit = True
        stopping.set()

    for sig in STOP_SIGNALS:
        with contextlib.suppress(ValueError, RuntimeError, NotImplementedError):  # not the main thread
            loop.add_signal_handler(sig, on_signal)
    tasks = [asyncio.create_task(s.serve(sockets=socks)) for s, socks in servers]
    signalled = asyncio.create_task(stopping.wait())
    try:
        await asyncio.wait([*tasks, signalled], return_when=asyncio.FIRST_COMPLETED)
    finally:
        try:
            # Targeted workspace stops need the phone websocket and its event
            # loop alive. Uvicorn closes connections before lifespan teardown.
            if shutdown:
                await asyncio.to_thread(shutdown)
        finally:
            signalled.cancel()
            await asyncio.gather(signalled, return_exceptions=True)
            for s, _ in servers:
                s.should_exit = True
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sig in STOP_SIGNALS:
                with contextlib.suppress(ValueError, RuntimeError, NotImplementedError):
                    loop.remove_signal_handler(sig)
    for r in results:
        if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
            raise r


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m rose.app", description="Rose v3 voice assistant server.")
    p.add_argument("--brain", default="local",
                   help="local (default: $ROSE_BRAIN or brain.REPO), an MLX model for Rose's conversation, "
                        "codex: GPT 6.1 Terra Fast via Sign in with ChatGPT, "
                        "or claude: Claude via the local Claude Code subscription login")
    p.add_argument("--transcript-dir", help="durable private transcript directory (Codex mode)")
    p.add_argument("--cloud-audio", help="Linux voice provider JSON with existing local model paths")
    p.add_argument("--pair-host", action="append", default=[], help="Tailscale MagicDNS/IPv4 host for the shared QR; repeatable")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--voice", default="zoey", help="Phonon voice name (see /api/state voices)")
    p.add_argument("--no-browser-audio", dest="browser_audio", action="store_false",
                   help="do not use the browser for audio; Rose's speech plays silently in real time")
    p.add_argument("--local-audio", action="store_true",
                   help="sound card mic and speaker via sounddevice (wear headphones)")
    p.add_argument("--input-device", default=None, help="sounddevice input (name or index)")
    p.add_argument("--output-device", default=None, help="sounddevice output (name or index)")
    p.add_argument("--keepalive", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the GPU queue warm while idle (default on)")
    p.add_argument("--ui-only", action="store_true", help="no models: replay a scripted demo session")
    p.add_argument("--phone", choices=("fake", "real"), default="fake",
                   help="real: control the Android phone running Rose Agent (USB: adb reverse tcp:8091 tcp:8091)")
    p.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT,
                   help="TLS listener on 0.0.0.0 for phones paired by QR code (Wi-Fi/Tailscale)")
    p.add_argument("--no-remote", dest="remote", action="store_false",
                   help="no remote listener: the phone can only use USB (adb reverse)")
    p.add_argument("--task-brain", default=None,
                   help="MLX model for the phone agent and thinker (default $ROSE_TASK_BRAIN, else --brain)")
    p.add_argument("--log-level", default="warning")
    args = p.parse_args(argv)
    if args.cloud_audio and args.brain not in ('codex', 'claude'):
        p.error('--cloud-audio requires --brain codex or claude')
    from .remote import configured_hosts
    try:
        args.pair_host = configured_hosts(args.pair_host)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = Server(args)
    remote = (Remote(port=args.remote_port, **({'hosts': lambda: args.pair_host} if args.pair_host else {}))
              if args.remote else None)
    servers = build_servers(server, args, remote)
    print(f"[rose] dashboard: http://127.0.0.1:{args.port}  (mode: {server.mode})", flush=True)
    if remote is not None and remote.listening:
        print(f"[rose] remote: https://0.0.0.0:{args.remote_port} for paired phones "
              f"(server id {remote.identity.server_id}; pair with the dashboard's Pair phone button "
              "or python -m rose.pair)", flush=True)
    try:
        asyncio.run(serve_all(servers, shutdown=server.shutdown))
    except KeyboardInterrupt:
        pass
    finally:
        if remote is not None:
            remote.flush()


if __name__ == "__main__":
    main()
