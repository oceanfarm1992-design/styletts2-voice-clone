"""
Engine workers. Each heavy engine (StyleTTS2, IndicF5, RVC, Kokoro) pins its own
incompatible set of libraries, so each one runs in its own virtualenv as a
long-lived subprocess: `python -m voiceclone.workers.<engine>`.

Protocol (one JSON object per line):
    parent -> worker stdin : {"text": ..., "out_wav": ..., ...engine kwargs}
    worker -> parent stdout: {"ok": true} or {"ok": false, "error": "..."}

The model is loaded once on the first request and reused for the rest of the
process, so narrating twenty scenes costs one model load, not twenty. Worker
logging goes to stderr only; stdout is reserved for protocol replies.
"""
