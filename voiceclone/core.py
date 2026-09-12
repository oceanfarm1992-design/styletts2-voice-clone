"""
Reusable StyleTTS2 voice-cloning TTS chain, with Kokoro-82M, Piper, and
espeak-ng fallbacks. Designed to run inside a CI job (e.g. a GitHub Actions
Ubuntu runner, CPU-only) as part of an automated video pipeline.

engine="auto" tries, in order:
  1. styletts2 - clones a voice from a reference sample fetched at runtime
     from a private GitHub repo (never bundled with this package or written
     to disk anywhere persistent). Needs a fine-grained, read-only GitHub PAT
     and the target repo path — see synthesize()'s voice_ref_repo /
     voice_repo_pat_env arguments.
  2. kokoro - open-weights neural TTS (hexgrad/Kokoro-82M), fully offline
     once its checkpoint is cached, not your voice.
  3. piper - fully-offline neural TTS, voice models cached locally, not your
     voice.
  4. espeak-ng - last resort; must be installed separately (e.g.
     `apt-get install espeak-ng` on Ubuntu). Worst quality, always available.

engine="openai" is also available for manual/one-off use (needs
OPENAI_API_KEY and the `openai` package) but isn't part of the "auto" chain
since it isn't a clone of your own voice.

Every per-engine function raises on failure; synthesize() catches that and
tries the next engine in the chain, so a single dependency/secret/network
problem never has to take down an otherwise-automated pipeline.
"""
import functools
import os
import subprocess
import sys

import requests

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

_KOKORO_PIPELINE = None
_KOKORO_LANG = None
_STYLETTS2_MODEL = None
_VOICE_REF_PATH = None

# HuggingFace raw file base for piper voices.
# Path layout: <lang>/<lang_region>/<name>/<quality>/<voice>.onnx[.json]
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICE_PATHS = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-ryan-high": "en/en_US/ryan/high/en_US-ryan-high.onnx",
}
HEADERS = {"User-Agent": "styletts2-voice-clone/0.1"}

# StyleTTS2's own long-text splitter chunks at 420 raw characters by default, but
# ALL-CAPS emphasis / "..." pauses common in AI-written scripts can phonemize to
# more tokens per character than that budget assumes, overflowing the underlying
# model's 512-token limit even within one "safe" chunk. Shrink it for headroom.
STYLETTS2_CHUNK_CHARS = 250


class AllEnginesFailedError(RuntimeError):
    """Raised by synthesize() only when every engine in the chain failed."""


# --------------------------------------------------------------------------- OpenAI
def run_openai(text: str, out_wav: str, model: str = "tts-1", voice: str = "onyx",
                instructions: str | None = None) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed.")

    client = OpenAI(api_key=api_key)
    kwargs = dict(model=model, voice=voice, input=text, response_format="wav")
    if instructions:
        kwargs["instructions"] = instructions
    with client.audio.speech.with_streaming_response.create(**kwargs) as response:
        response.stream_to_file(out_wav)


# ---------------------------------------------------------------------------- Kokoro
def _get_kokoro_pipeline(lang_code: str):
    global _KOKORO_PIPELINE, _KOKORO_LANG
    if _KOKORO_PIPELINE is None or _KOKORO_LANG != lang_code:
        from kokoro import KPipeline
        _KOKORO_PIPELINE = KPipeline(lang_code=lang_code)
        _KOKORO_LANG = lang_code
    return _KOKORO_PIPELINE


def run_kokoro(text: str, out_wav: str, voice: str = "am_fenrir", lang_code: str = "a") -> None:
    import numpy as np
    import soundfile as sf

    pipeline = _get_kokoro_pipeline(lang_code)
    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    sf.write(out_wav, full, 24000)


# ------------------------------------------------------------------------- StyleTTS2
def _fetch_voice_reference(repo: str, file: str, pat: str, cache_path: str) -> str:
    """Download the private reference voice clip at runtime via a fine-grained,
    read-only PAT scoped to a separate private repo. Never committed anywhere —
    the clip is personal biometric-ish data and this package is public."""
    global _VOICE_REF_PATH
    if _VOICE_REF_PATH and os.path.exists(_VOICE_REF_PATH):
        return _VOICE_REF_PATH
    if not repo:
        raise RuntimeError(
            "No voice reference repo configured (pass voice_ref_repo= or set VOICE_REF_REPO)."
        )
    if not pat:
        raise RuntimeError(
            "No PAT available to fetch the voice reference sample "
            "(set the env var named by voice_repo_pat_env)."
        )
    url = f"https://api.github.com/repos/{repo}/contents/{file}"
    headers = {"Authorization": f"Bearer {pat}", "Accept": "application/vnd.github.raw+json"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(resp.content)
    _VOICE_REF_PATH = cache_path
    return cache_path


def _get_styletts2_model():
    global _STYLETTS2_MODEL
    if _STYLETTS2_MODEL is None:
        import nltk
        import torch

        # styletts2's TextCleaner debug-prints raw phoneme text (including rare IPA
        # characters) on any symbol outside its vocabulary — harmless on Linux CI
        # (UTF-8 locale) but crashes on Windows consoles (cp1252). Widen stdout
        # defensively so local runs behave the same as CI.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        # styletts2's bundled checkpoint loader calls torch.load() without
        # weights_only=False. PyTorch >=2.6 defaults weights_only=True, which
        # rejects this (older, trusted, official StyleTTS2/LibriTTS) checkpoint
        # format. Patch the default rather than editing the installed package.
        torch.load = functools.partial(torch.load, weights_only=False)
        nltk.download("punkt_tab", quiet=True)

        import styletts2.tts as _styletts2_tts
        _styletts2_tts.SINGLE_INFERENCE_MAX_LEN = STYLETTS2_CHUNK_CHARS

        from styletts2.tts import StyleTTS2
        _STYLETTS2_MODEL = StyleTTS2()
    return _STYLETTS2_MODEL


def run_styletts2(text: str, out_wav: str, voice_ref_repo: str | None,
                   voice_ref_file: str = "reference_voice.mp3",
                   pat_env: str = "VOICE_REPO_PAT",
                   cache_path: str = "build/.voice_reference.mp3") -> None:
    pat = os.environ.get(pat_env, "").strip()
    ref_path = _fetch_voice_reference(voice_ref_repo, voice_ref_file, pat, cache_path)
    model = _get_styletts2_model()
    model.inference(text, target_voice_path=ref_path, output_wav_file=out_wav)


# ----------------------------------------------------------------------------- Piper
def _download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _ensure_piper_voice(voice: str, voices_dir: str) -> str:
    if voice not in VOICE_PATHS:
        raise RuntimeError(f"Unknown Piper voice '{voice}'. Known: {list(VOICE_PATHS)}")
    os.makedirs(voices_dir, exist_ok=True)
    onnx_rel = VOICE_PATHS[voice]
    onnx_path = os.path.join(voices_dir, os.path.basename(onnx_rel))
    _download(f"{HF_BASE}/{onnx_rel}", onnx_path)
    _download(f"{HF_BASE}/{onnx_rel}.json", onnx_path + ".json")
    return onnx_path


def run_piper(text: str, out_wav: str, voice: str = "en_US-amy-medium",
              voices_dir: str = "voices", length_scale: float = 1.0,
              sentence_silence: float = 0.3) -> None:
    onnx_path = _ensure_piper_voice(voice, voices_dir)
    cmd = ["piper", "--model", onnx_path, "--output_file", out_wav,
           "--length_scale", str(length_scale),
           "--sentence_silence", str(sentence_silence)]
    proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"piper failed: {proc.stderr.decode('utf-8', 'replace')}")


# ---------------------------------------------------------------------------- espeak
def run_espeak(text: str, out_wav: str, voice: str = "en-us+m3", speed: int = 135,
               pitch: int = 45, gap: int = 6) -> None:
    cmd = ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-g", str(gap),
           "-w", out_wav, text]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {proc.stderr.decode('utf-8', 'replace')}")


# ------------------------------------------------------------------------------ main
def synthesize(
    text: str,
    out_wav: str,
    engine: str = "auto",
    voice_ref_repo: str | None = None,
    voice_ref_file: str = "reference_voice.mp3",
    voice_repo_pat_env: str = "VOICE_REPO_PAT",
    voice_ref_cache: str = "build/.voice_reference.mp3",
    kokoro_voice: str = "am_fenrir",
    kokoro_lang: str = "a",
    piper_voice: str = "en_US-amy-medium",
    voices_dir: str = "voices",
    openai_model: str = "tts-1",
    openai_voice: str = "onyx",
    openai_instructions: str | None = None,
) -> str:
    """Synthesize `text` to `out_wav`.

    Returns the name of the engine that actually produced the audio.

    engine="auto" (the default) tries styletts2 -> kokoro -> piper -> espeak,
    moving to the next engine on any failure (missing secret, network error,
    missing binary, etc). Pass a specific engine name to skip the chain and
    use only that one. Raises AllEnginesFailedError only when every engine
    tried has failed.
    """
    engines = [engine] if engine != "auto" else ["styletts2", "kokoro", "piper", "espeak"]
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)

    last_error: Exception | None = None
    for eng in engines:
        try:
            if eng == "styletts2":
                run_styletts2(text, out_wav, voice_ref_repo, voice_ref_file,
                              voice_repo_pat_env, voice_ref_cache)
            elif eng == "kokoro":
                run_kokoro(text, out_wav, kokoro_voice, kokoro_lang)
            elif eng == "piper":
                run_piper(text, out_wav, piper_voice, voices_dir)
            elif eng == "espeak":
                run_espeak(text, out_wav)
            elif eng == "openai":
                run_openai(text, out_wav, openai_model, openai_voice, openai_instructions)
            else:
                raise ValueError(f"Unknown engine '{eng}'")

            if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
                return eng
        except Exception as exc:  # noqa: BLE001 — try the next engine
            last_error = exc
            print(f"[voiceclone] {eng} failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    raise AllEnginesFailedError(f"All TTS engines failed. Last error: {last_error}")
