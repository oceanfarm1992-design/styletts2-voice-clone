"""Kokoro-82M worker: fixed (not cloned) neural voices, used as a fallback base
voice that RVC then converts into the trained voice.

Request: {"text", "out_wav", "voice", "lang_code"}
"""
from ._serve import serve

_PIPELINES: dict = {}


def handle(req: dict) -> None:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    lang = req.get("lang_code", "a")
    if lang not in _PIPELINES:
        _PIPELINES[lang] = KPipeline(lang_code=lang)
    chunks = [audio for _, _, audio in _PIPELINES[lang](req["text"], voice=req.get("voice", "am_fenrir"))]
    if not chunks:
        raise ValueError("Kokoro produced no audio.")
    sf.write(req["out_wav"], np.concatenate(chunks), 24000)


if __name__ == "__main__":
    serve(handle)
