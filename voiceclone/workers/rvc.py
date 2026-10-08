"""RVC worker: converts any speech (StyleTTS2, IndicF5, Edge, Kokoro...) into the
trained voice, using Applio's inference code.

Request: {"in_wav", "out_wav", "model_path", "index_path", "pitch", "index_rate",
          "protect", "f0_method"}

Needs the APPLIO_DIR env var: an Applio checkout with rvc/models/predictors/
rmvpe.pt and rvc/models/embedders/contentvec/ in place. Applio resolves its
config files relative to the working directory, so the worker runs from there.
"""
import os
import sys

from ._serve import serve

# Relative request paths mean relative to where the worker was started; the
# worker itself moves into APPLIO_DIR on its first request.
START_DIR = os.getcwd()
_CONVERTER = None


def _converter():
    global _CONVERTER
    if _CONVERTER is None:
        applio = os.environ.get("APPLIO_DIR", "")
        if not os.path.isdir(os.path.join(applio, "rvc")):
            raise RuntimeError(f"APPLIO_DIR is not an Applio checkout: {applio!r}")
        os.chdir(applio)
        sys.path.insert(0, applio)
        from rvc.infer.infer import VoiceConverter
        _CONVERTER = VoiceConverter()
    return _CONVERTER


def handle(req: dict) -> None:
    paths = {k: os.path.join(START_DIR, req[k])
             for k in ("in_wav", "out_wav", "model_path", "index_path")}
    _converter().convert_audio(
        audio_input_path=paths["in_wav"],
        audio_output_path=paths["out_wav"],
        model_path=paths["model_path"],
        index_path=paths["index_path"],
        pitch=int(req.get("pitch", 0)),
        f0_method=req.get("f0_method", "rmvpe"),
        index_rate=float(req.get("index_rate", 0.75)),
        protect=float(req.get("protect", 0.5)),
        volume_envelope=1.0,
        split_audio=False,
        export_format="WAV",
        embedder_model="contentvec",
    )
    if not os.path.exists(paths["out_wav"]):
        raise RuntimeError("Applio finished without writing the output file.")


if __name__ == "__main__":
    serve(handle)
