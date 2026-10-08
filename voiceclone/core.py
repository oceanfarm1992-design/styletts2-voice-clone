"""
One voice engine for every video pipeline: speaks English and Tamil in a
cloned voice, on CPU, inside a CI job (built for GitHub Actions).

Two stages:
  1. Base speech, first engine in the language's chain that succeeds:
       en: styletts2 -> kokoro -> piper -> edge -> espeak
       ta: indicf5   -> edge   -> espeak
     styletts2 and indicf5 clone the voice (and speaking style) from reference
     clips; the rest are fixed stock voices.
  2. RVC voice conversion with a model trained on your own voice (on a local
     GPU, see README), so even a stock-voice fallback comes out in your voice.

Reference clips, the RVC model and profile.json (see profile.py) are fetched
at runtime from a separate private repo. Heavy engines run as long-lived
workers in their own virtualenvs (see pool.py). Every engine raises on
failure; synthesize() moves to the next one, so a missing secret or a flaky
download never takes down an otherwise-automated pipeline.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import requests

from . import pool
from .profile import Profile, ProfileError, VoiceSource

LANGUAGE_CHAINS = {
    "en": ["styletts2", "kokoro", "piper", "edge", "espeak"],
    "ta": ["indicf5", "edge", "espeak"],
}
CLONING_ENGINES = {"styletts2", "indicf5"}
ALL_ENGINES = sorted({e for chain in LANGUAGE_CHAINS.values() for e in chain} | {"openai"})

EDGE_VOICES = {"en": "en-US-GuyNeural", "ta": "ta-LK-KumarNeural"}
ESPEAK_VOICES = {"en": "en-us+m3", "ta": "ta"}

# HuggingFace raw file base for piper voices.
# Path layout: <lang>/<lang_region>/<name>/<quality>/<voice>.onnx[.json]
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICE_PATHS = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-ryan-high": "en/en_US/ryan/high/en_US-ryan-high.onnx",
}
HEADERS = {"User-Agent": "styletts2-voice-clone/0.3"}

_SOURCES: dict = {}


class AllEnginesFailedError(RuntimeError):
    """Raised by synthesize() only when every engine in the chain failed."""


def _log(msg: str) -> None:
    print(f"[voiceclone] {msg}", file=sys.stderr)


# ---------------------------------------------------------------- cloning engines
def run_styletts2(text: str, out_wav: str, source: VoiceSource, settings: dict) -> None:
    if not settings.get("voice"):
        raise ProfileError("No English voice reference in the profile.")
    style = settings.get("style")
    pool.worker("styletts2").request({
        "text": text, "out_wav": os.path.abspath(out_wav),
        "voice_path": source.fetch(settings["voice"]),
        "style_path": source.fetch(style) if style else None,
        "alpha": settings.get("alpha", 0.3), "beta": settings.get("beta"),
        "embedding_scale": settings.get("embedding_scale", 1.0),
    })


def run_indicf5(text: str, out_wav: str, source: VoiceSource, settings: dict) -> None:
    if not (settings.get("ref_audio") and settings.get("ref_text")):
        raise ProfileError("IndicF5 needs 'ref_audio' and its exact 'ref_text' in the profile.")
    pool.worker("indicf5").request({
        "text": text, "out_wav": os.path.abspath(out_wav),
        "ref_audio": source.fetch(settings["ref_audio"]), "ref_text": settings["ref_text"],
        "nfe_step": settings.get("nfe_step"), "speed": settings.get("speed"),
    })


# ------------------------------------------------------------ stock-voice engines
def run_kokoro(text: str, out_wav: str, voice: str = "am_fenrir", lang_code: str = "a") -> None:
    pool.worker("kokoro").request({"text": text, "out_wav": os.path.abspath(out_wav),
                                   "voice": voice, "lang_code": lang_code})


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


def _to_wav(src: str, out_wav: str) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, out_wav],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')}")


def run_edge(text: str, out_wav: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
    """Microsoft Edge online neural TTS (needs network, no key)."""
    import asyncio

    import edge_tts

    with tempfile.TemporaryDirectory() as tmp:
        mp3 = os.path.join(tmp, "edge.mp3")
        asyncio.run(edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(mp3))
        _to_wav(mp3, out_wav)


def run_espeak(text: str, out_wav: str, voice: str = "en-us+m3", speed: int = 135,
               pitch: int = 45, gap: int = 6) -> None:
    cmd = ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-g", str(gap),
           "-w", out_wav, text]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {proc.stderr.decode('utf-8', 'replace')}")


def run_openai(text: str, out_wav: str, model: str = "tts-1", voice: str = "onyx",
               instructions: str | None = None) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from openai import OpenAI

    kwargs = dict(model=model, voice=voice, input=text, response_format="wav")
    if instructions:
        kwargs["instructions"] = instructions
    with OpenAI(api_key=api_key).audio.speech.with_streaming_response.create(**kwargs) as response:
        response.stream_to_file(out_wav)


# --------------------------------------------------------------------------- RVC
def run_rvc(in_wav: str, out_wav: str, source: VoiceSource, profile: Profile) -> None:
    rvc = profile.rvc
    if rvc is None:
        raise ProfileError("No 'rvc' model in the profile.")
    pool.worker("rvc").request({
        "in_wav": os.path.abspath(in_wav), "out_wav": os.path.abspath(out_wav),
        "model_path": source.fetch(rvc.model), "index_path": source.fetch(rvc.index),
        "pitch": rvc.pitch, "index_rate": rvc.index_rate, "protect": rvc.protect,
        "f0_method": "rmvpe",
    })


# -------------------------------------------------------------------------- main
def _source(repo: str | None, pat_env: str, cache_dir: str) -> VoiceSource:
    key = (repo, pat_env, cache_dir, os.environ.get("VOICE_REF_DIR"))
    if key not in _SOURCES:
        _SOURCES[key] = VoiceSource.from_env(repo, pat_env, cache_dir)
    return _SOURCES[key]


def _load_profile(source: VoiceSource, voice_ref_file: str, style_ref_file: str | None) -> Profile:
    try:
        return source.profile(voice_ref_file, style_ref_file)
    except ProfileError as exc:
        _log(f"voice profile unavailable ({exc}); only stock voices will work")
        return Profile()


def _has_audio(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def synthesize(
    text: str,
    out_wav: str,
    engine: str = "auto",
    language: str = "en",
    style: str = "default",
    rvc: str = "auto",
    voice_ref_repo: str | None = None,
    voice_ref_file: str = "reference_voice.mp3",
    voice_repo_pat_env: str = "VOICE_REPO_PAT",
    voice_ref_cache: str = "build/.voice_reference.mp3",
    style_ref_file: str | None = None,
    styletts2_alpha: float | None = None,
    styletts2_beta: float | None = None,
    styletts2_embedding_scale: float | None = None,
    kokoro_voice: str = "am_fenrir",
    kokoro_lang: str = "a",
    piper_voice: str = "en_US-amy-medium",
    edge_voice: str | None = None,
    voices_dir: str = "voices",
    openai_model: str = "tts-1",
    openai_voice: str = "onyx",
    openai_instructions: str | None = None,
) -> str:
    """Synthesize `text` (in `language`, "en" or "ta") to `out_wav`.

    Returns what produced the audio, e.g. "styletts2+rvc" or "edge+rvc".

    engine="auto" walks the language's chain (see LANGUAGE_CHAINS), moving on
    at any failure; a specific engine name runs only that one. style picks a
    style block from the voice profile ("default", "comedy", ...).

    rvc: "auto" converts the result with the profile's RVC model when there is
    one and keeps the unconverted audio if conversion fails; "on" treats a
    conversion failure as that engine failing; "off" skips it.

    style_ref_file / styletts2_* override the profile's English settings
    (kept for callers written against 0.2). Raises AllEnginesFailedError only
    when every engine tried has failed.
    """
    if language not in LANGUAGE_CHAINS:
        raise ValueError(f"Unsupported language '{language}'. Known: {list(LANGUAGE_CHAINS)}")
    if rvc not in ("auto", "on", "off"):
        raise ValueError("rvc must be 'auto', 'on' or 'off'")
    engines = LANGUAGE_CHAINS[language] if engine == "auto" else [engine]
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)

    source = _source(voice_ref_repo, voice_repo_pat_env,
                     os.path.join(os.path.dirname(voice_ref_cache) or ".", ".voice_refs"))
    profile = _load_profile(source, voice_ref_file, style_ref_file)
    settings = profile.style_for(style, language)
    overrides = {"style": style_ref_file, "alpha": styletts2_alpha, "beta": styletts2_beta,
                 "embedding_scale": styletts2_embedding_scale}
    if language == "en":
        settings.update({k: v for k, v in overrides.items() if v is not None})
        settings.setdefault("voice", voice_ref_file)

    use_rvc = rvc != "off" and profile.rvc is not None
    if rvc == "on" and profile.rvc is None:
        raise AllEnginesFailedError("rvc='on' but the voice profile has no RVC model.")

    last_error: Exception | None = None
    with tempfile.TemporaryDirectory() as tmp:
        base_wav = os.path.join(tmp, "base.wav") if use_rvc else out_wav
        for eng in engines:
            try:
                _run_base(eng, text, base_wav, language, source, settings, kokoro_voice,
                          kokoro_lang, piper_voice, edge_voice, voices_dir,
                          openai_model, openai_voice, openai_instructions)
                if not _has_audio(base_wav):
                    raise RuntimeError("engine finished without writing audio")
                if not use_rvc:
                    return eng
                return eng + _convert(base_wav, out_wav, source, profile, strict=rvc == "on")
            except Exception as exc:  # noqa: BLE001 - try the next engine
                last_error = exc
                _log(f"{eng} failed: {type(exc).__name__}: {exc}")
    raise AllEnginesFailedError(f"All TTS engines failed. Last error: {last_error}")


def _convert(base_wav: str, out_wav: str, source: VoiceSource, profile: Profile,
             strict: bool) -> str:
    """RVC-convert base_wav into out_wav; returns the engine-name suffix."""
    try:
        run_rvc(base_wav, out_wav, source, profile)
        if _has_audio(out_wav):
            return "+rvc"
        raise RuntimeError("RVC finished without writing audio")
    except Exception as exc:  # noqa: BLE001 - keep the unconverted audio unless strict
        if strict:
            raise
        _log(f"rvc failed, keeping unconverted audio: {type(exc).__name__}: {exc}")
        shutil.copyfile(base_wav, out_wav)
        return ""


def _run_base(eng, text, out_wav, language, source, settings, kokoro_voice, kokoro_lang,
              piper_voice, edge_voice, voices_dir, openai_model, openai_voice,
              openai_instructions) -> None:
    if eng == "styletts2":
        run_styletts2(text, out_wav, source, settings)
    elif eng == "indicf5":
        run_indicf5(text, out_wav, source, settings)
    elif eng == "kokoro":
        run_kokoro(text, out_wav, kokoro_voice, kokoro_lang)
    elif eng == "piper":
        run_piper(text, out_wav, piper_voice, voices_dir)
    elif eng == "edge":
        run_edge(text, out_wav, edge_voice or EDGE_VOICES[language])
    elif eng == "espeak":
        run_espeak(text, out_wav, ESPEAK_VOICES[language])
    elif eng == "openai":
        run_openai(text, out_wav, openai_model, openai_voice, openai_instructions)
    else:
        raise ValueError(f"Unknown engine '{eng}'")
