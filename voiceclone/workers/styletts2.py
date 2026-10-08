"""StyleTTS2 worker (English): clones timbre from one clip, prosody from another.

Request: {"text", "out_wav", "voice_path", "style_path"?, "alpha", "beta"?,
          "embedding_scale"}
"""
import functools
import sys

from ._serve import serve

# StyleTTS2's own long-text splitter chunks at 420 raw characters by default, but
# ALL-CAPS emphasis / "..." pauses common in AI-written scripts can phonemize to
# more tokens per character than that budget assumes, overflowing the underlying
# model's 512-token limit even within one "safe" chunk. Shrink it for headroom.
STYLETTS2_CHUNK_CHARS = 250

# Phoneme-level pronunciation fixes, applied after StyleTTS2's own phonemizer.
# "GitHub" phonemizes to ɡˈɪthʌb, and the model stretches that /h/ into a ~0.5s
# near-silent breath ("Git ... Hub"); dropping it gives the fast spoken "Git-ub".
PHONEME_FIXES = {
    "ɡˈɪthʌb": "ɡˈɪtʌb",
}

_MODEL = None
_STYLE_VECTORS: dict = {}  # (voice path, style path) -> spliced style tensor


class _FixedPhonemizer:
    def __init__(self, inner):
        self.inner = inner

    def phonemize(self, text):
        out = self.inner.phonemize(text)
        for src, dst in PHONEME_FIXES.items():
            out = out.replace(src, dst)
        return out


def _model():
    global _MODEL
    if _MODEL is None:
        import nltk
        import torch

        # styletts2's TextCleaner debug-prints raw phoneme text (including rare IPA
        # characters); widen stderr so Windows consoles (cp1252) don't crash on it.
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

        # styletts2's bundled checkpoint loader calls torch.load() without
        # weights_only=False. PyTorch >=2.6 defaults weights_only=True, which
        # rejects this (older, trusted, official StyleTTS2/LibriTTS) checkpoint.
        torch.load = functools.partial(torch.load, weights_only=False)
        nltk.download("punkt_tab", quiet=True)

        import styletts2.tts as _styletts2_tts
        _styletts2_tts.SINGLE_INFERENCE_MAX_LEN = STYLETTS2_CHUNK_CHARS

        from styletts2.tts import StyleTTS2
        _MODEL = StyleTTS2()
        _MODEL.phoneme_converter = _FixedPhonemizer(_MODEL.phoneme_converter)
    return _MODEL


def _style_vector(model, voice_path: str, style_path: str | None):
    """StyleTTS2's 256-dim style vector is [timbre (128) | prosody (128)]:
    style_encoder -> who it sounds like, predictor_encoder -> how it is spoken
    (pacing, intonation, energy). With a separate style clip, keep the timbre of
    the voice sample and take the prosody from the style clip."""
    key = (voice_path, style_path)
    if key not in _STYLE_VECTORS:
        voice = model.compute_style(voice_path)
        if style_path:
            import torch
            style = model.compute_style(style_path)
            voice = torch.cat([voice[:, :128], style[:, 128:]], dim=1)
        _STYLE_VECTORS[key] = voice
    return _STYLE_VECTORS[key]


def handle(req: dict) -> None:
    style_path = req.get("style_path")
    beta = req.get("beta")
    if beta is None:
        beta = 0.0 if style_path else 0.7
    model = _model()
    model.inference(req["text"], output_wav_file=req["out_wav"],
                    ref_s=_style_vector(model, req["voice_path"], style_path),
                    alpha=req.get("alpha", 0.3), beta=beta,
                    embedding_scale=req.get("embedding_scale", 1.0))


if __name__ == "__main__":
    serve(handle)
