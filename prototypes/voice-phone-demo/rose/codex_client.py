"""Thread-safe stdio client of the existing Node Codex adapter; no second app-server client."""
from __future__ import annotations

import concurrent.futures
import itertools
import json
import os
from pathlib import Path
import subprocess
import threading


class CodexClient:
    def __init__(self, on_event, command=None):
        root = Path(__file__).resolve().parents[3]
        self.cwd = root / 'workspace'
        self.on_event = on_event
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.pending = {}
        self.closed = False
        self.process = subprocess.Popen(command or [os.environ.get('ROSE_NODE_BIN', 'node'),
            str(self.cwd / 'rose-agent/server/codex-voice-bridge.mjs')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', errors='strict', bufsize=1)  # the bridge writes UTF-8 whatever the host locale
        self.reader = threading.Thread(target=self._read, name='codex-events', daemon=True)
        self.reader.start()

    def send(self, message):
        with self.lock:
            if self.closed:
                raise RuntimeError('Codex bridge stopped')
            self.process.stdin.write(json.dumps(message) + '\n')
            self.process.stdin.flush()

    def request(self, op, session, **params):
        fid = next(self.ids)
        future = concurrent.futures.Future()
        with self.lock:
            self.pending[fid] = future
        try:
            self.send(dict(id=fid, op=op, session=session, **params))
            return future.result(timeout=100)
        finally:
            with self.lock:
                self.pending.pop(fid, None)

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if 'id' in m:
                    with self.lock:
                        fut = self.pending.pop(m['id'], None)
                    if fut and not fut.done():
                        if m.get('error'):
                            fut.set_exception(RuntimeError(m['error']))
                        else:
                            fut.set_result(m.get('result', {}))
                else:
                    self.on_event(m)
        finally:
            with self.lock:
                self.closed = True
                for fut in self.pending.values():
                    if not fut.done():
                        fut.set_exception(RuntimeError('Codex bridge stopped'))
                self.pending.clear()
            self.on_event({'event': 'bridge_failure'})

    def close(self):
        with self.lock:
            if not self.closed:
                self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        self.process.stdout.close()


def ClaudeClient(on_event):
    """The same stdio protocol, served by Claude through the local Claude Code subscription login."""
    root = Path(__file__).resolve().parents[3]
    bridge = root / 'workspace' / 'rose-agent' / 'server' / 'claude' / 'claude-voice-bridge.mjs'
    return CodexClient(on_event, command=[os.environ.get('ROSE_BUN_BIN', 'bun'), str(bridge)])
