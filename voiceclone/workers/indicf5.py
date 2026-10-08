"""IndicF5 worker (Tamil and 10 other Indian languages): zero-shot cloning that
copies both voice and speaking tone from a short reference clip.

Request: {"text", "out_wav", "ref_audio", "ref_text", "nfe_step"?, "speed"?}

The model (ai4bharat/IndicF5) is gated on Hugging Face: the HF_TOKEN env var
must belong to an account that has accepted its terms.
"""
from ._serve import serve

REPO_ID = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000
# Flow-matching steps per chunk; cost is linear in this. The model's default 32
# is twice as slow as 16 at similar quality; 8 starts slurring words. (Dynamic
# int8 quantization was tried too: ~15% faster, but it also slurred words.)
# Every step re-reads the reference clip, so a 5-6 s clip is ~40% faster than
# an 11 s one at the same quality.
DEFAULT_NFE_STEPS = 16

_MODEL = None
_REFS: dict = {}  # (path, text) -> preprocessed (ref_audio, ref_text)


def _load():
    from transformers import AutoModel

    model = AutoModel.from_pretrained(REPO_ID, trust_remote_code=True)
    # The remote model code wraps the DiT and vocoder in torch.compile(), and
    # its checkpoint keys carry the wrapper's "_orig_mod." prefix, so the
    # wrapping must stay in place while loading. Compiling itself only
    # happens on the first call; on CPU it costs minutes, recompiles for
    # every new chunk length and needs a C++ toolchain, so unwrap and run
    # the (already loaded) eager modules instead.
    for name in ("ema_model", "vocoder"):
        module = getattr(model, name)
        setattr(model, name, getattr(module, "_orig_mod", module))
    return model


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load()
    return _MODEL


def handle(req: dict) -> None:
    import soundfile as sf
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    model = _model()
    key = (req["ref_audio"], req["ref_text"])
    if key not in _REFS:
        # Trims the clip to <15 s and normalises the transcript's punctuation.
        _REFS[key] = preprocess_ref_audio_text(req["ref_audio"], req["ref_text"],
                                               show_info=lambda *_: None)
    ref_audio, ref_text = _REFS[key]
    # infer_process chunks the text so reference + chunk stay under ~25 s,
    # and cross-fades the chunks back together.
    audio, sample_rate, _ = infer_process(
        ref_audio, ref_text, req["text"], model.ema_model, model.vocoder,
        mel_spec_type="vocos", nfe_step=int(req.get("nfe_step") or DEFAULT_NFE_STEPS),
        speed=float(req.get("speed") or 1.0), device=model.device,
    )
    sf.write(req["out_wav"], audio, sample_rate or SAMPLE_RATE)


if __name__ == "__main__":
    serve(handle)
