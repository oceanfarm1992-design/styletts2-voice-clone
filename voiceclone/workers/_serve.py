"""Line-delimited JSON request loop shared by every engine worker."""
import json
import os
import sys
import traceback
from typing import Any, Callable


def _private_reply_stream():
    """Keep the real stdout for protocol replies only, and point fd 1 (and
    sys.stdout) at stderr, so nothing a library prints - Python progress bars,
    native code, child processes like ffmpeg - can corrupt the reply stream."""
    sys.stdout.flush()
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return replies


def serve(handle: Callable[[dict], Any]) -> None:
    """Answer one JSON request per stdin line until EOF."""
    replies = _private_reply_stream()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            handle(json.loads(line))
            reply = {"ok": True}
        except Exception as exc:  # noqa: BLE001 - reported back to the parent
            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        replies.write(json.dumps(reply) + "\n")
        replies.flush()
