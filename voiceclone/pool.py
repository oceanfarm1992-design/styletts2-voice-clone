"""Long-lived engine worker subprocesses, one per engine, each in its own venv.

`setup_engines.sh` (run by the GitHub Action) creates
$VOICECLONE_HOME/venvs/<engine>/ for every engine. When an engine has no venv
there, its worker runs on the current interpreter instead, which works when the
engine's libraries are installed alongside this package (the pre-0.3 layout).
"""
import atexit
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

ENGINE_WORKERS = ("styletts2", "indicf5", "rvc", "kokoro")
# Per request. IndicF5 on a 4-core CI runner needs minutes per sentence, so a
# long Tamil script can take most of an hour; override with VOICECLONE_TIMEOUT.
DEFAULT_TIMEOUT_S = 3600


def request_timeout() -> float:
    try:
        return float(os.environ.get("VOICECLONE_TIMEOUT") or DEFAULT_TIMEOUT_S)
    except ValueError:
        return DEFAULT_TIMEOUT_S


def voiceclone_home() -> Path:
    return Path(os.environ.get("VOICECLONE_HOME") or Path.home() / ".voiceclone")


def python_for(engine: str) -> str:
    venv = voiceclone_home() / "venvs" / engine
    for candidate in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


class WorkerError(RuntimeError):
    """The worker reported a failure, died, or timed out."""


class Worker:
    def __init__(self, engine: str):
        if engine not in ENGINE_WORKERS:
            raise ValueError(f"Unknown engine worker '{engine}'")
        self.engine = engine
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()

    def _start(self) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self._proc = subprocess.Popen(
            [python_for(self.engine), "-m", f"voiceclone.workers.{self.engine}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", env=env,
        )
        self._replies = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, self._replies), daemon=True).start()

    @staticmethod
    def _pump(proc: subprocess.Popen, replies: queue.Queue) -> None:
        for line in proc.stdout:
            replies.put(line)
        replies.put(None)  # EOF: the worker exited

    def request(self, payload: dict, timeout: float | None = None) -> None:
        timeout = timeout or request_timeout()
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            self.close()
            raise WorkerError(f"{self.engine} worker is not accepting requests: {exc}") from exc
        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise WorkerError(f"{self.engine} worker timed out after {timeout:.0f}s") from None
        if line is None:
            code = self._proc.wait()
            self._proc = None
            raise WorkerError(f"{self.engine} worker exited (code {code}); see stderr above")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise WorkerError(f"{self.engine}: {reply.get('error', 'unknown error')}")

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        self._proc = None


_WORKERS: dict[str, Worker] = {}


def worker(engine: str) -> Worker:
    if engine not in _WORKERS:
        _WORKERS[engine] = Worker(engine)
    return _WORKERS[engine]


@atexit.register
def close_all() -> None:
    for w in _WORKERS.values():
        w.close()
