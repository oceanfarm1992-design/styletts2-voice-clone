import json
import subprocess
import sys

import pytest

from voiceclone import pool


def test_python_for_prefers_engine_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICECLONE_HOME", str(tmp_path))
    assert pool.python_for("rvc") == sys.executable
    exe = tmp_path / "venvs" / "rvc" / "bin" / "python"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert pool.python_for("rvc") == str(exe)


def test_unknown_worker_rejected():
    with pytest.raises(ValueError):
        pool.Worker("nope")


def _serve_echo(lines: list[dict]) -> list[dict]:
    """Run the shared serve loop with a handler that fails on {"fail": true}."""
    code = ("import os, subprocess, sys; from voiceclone.workers._serve import serve\n"
            "def h(r):\n"
            "    print('noise that must not reach the reply stream')\n"
            "    os.write(1, b'native noise\\n')\n"
            "    subprocess.run([sys.executable, '-c', 'print(1)'])\n"
            "    if r.get('fail'): raise ValueError('boom')\n"
            "serve(h)\n")
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True,
                          input="".join(json.dumps(x) + "\n" for x in lines))
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_serve_replies_once_per_request_and_reports_errors():
    replies = _serve_echo([{"a": 1}, {"fail": True}, {"b": 2}])
    assert replies == [{"ok": True}, {"ok": False, "error": "ValueError: boom"}, {"ok": True}]
