# styletts2-voice-clone

A reusable voiceover-generation chain for CI-driven video pipelines (built for
GitHub Actions, but not tied to it): clone your own voice with **StyleTTS2**,
falling back through **Kokoro-82M** → **Piper** → `espeak-ng` if the clone step
fails for any reason. Extracted from a personal daily-short-video pipeline so
multiple video-generation repos can share one TTS implementation instead of
each carrying its own copy.

This repo holds only the *code*. Your actual voice sample stays wherever you
already keep it — **never commit a voice sample here**. The intended pattern
is a separate **private** GitHub repo holding one reference audio clip, fetched
at runtime via a read-only token. See [Voice reference setup](#voice-reference-setup).

## Install

Straight from GitHub, no PyPI package needed:

```bash
pip install "styletts2-voice-clone[all] @ git+https://github.com/oceanfarm1992-design/styletts2-voice-clone.git"
```

`[all]` pulls in every engine's dependencies (StyleTTS2 needs `torch` —
install the CPU-only wheel *first* in CI, see below, or it'll pull the much
larger CUDA build). Use `[styletts2]`, `[kokoro]`, or `[openai]` individually
if you only want specific engines; the bare package (no extras) only needs
`requests` and can still run `espeak-ng` if that's on the system `PATH`.

## Voice reference setup

1. Create a **private** GitHub repo containing one short (10-30s is plenty)
   clean audio sample of the voice to clone, e.g. `reference_voice.mp3`.
2. Create a **fine-grained personal access token** scoped read-only to just
   that repo (Settings → Developer settings → Fine-grained tokens).
   **Never commit this token or paste it into chat with anyone/anything** —
   store it as a secret in whatever CI system calls this package.
3. Pass the repo path and token via arguments/env vars (see below).

## CLI usage

```bash
export VOICE_REF_REPO="you/voice-reference-audio"
export VOICE_REPO_PAT="<fine-grained read-only PAT>"

voiceclone --script build/script.txt --out build/voice.wav
# -> tries styletts2 (your cloned voice) -> kokoro -> piper -> espeak

voiceclone --engine kokoro --voice am_fenrir     # skip the clone, force an engine
voiceclone --engine piper --voice en_US-ryan-high
voiceclone --engine espeak
```

All options: `voiceclone --help`.

## Python API

```python
from voiceclone import synthesize

engine_used = synthesize(
    "Text to narrate goes here.",
    "build/voice.wav",
    voice_ref_repo="you/voice-reference-audio",   # or rely on VOICE_REF_REPO env var
)
print(f"synthesized with: {engine_used}")
```

`synthesize()` raises `AllEnginesFailedError` only if every engine in the
chain fails — a missing secret or a flaky network call falls through to the
next engine rather than raising immediately.

## GitHub Actions example

```yaml
- name: Install python deps
  run: |
    pip install torch==2.8.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
    pip install "styletts2-voice-clone[all] @ git+https://github.com/oceanfarm1992-design/styletts2-voice-clone.git"

- name: Generate voiceover
  env:
    VOICE_REF_REPO: you/voice-reference-audio
    VOICE_REPO_PAT: ${{ secrets.VOICE_REPO_PAT }}
  run: voiceclone --script build/script.txt --out build/voice.wav
```

## Notes

- StyleTTS2 (the underlying model here) is **English only** — it was trained
  on English speech (LJSpeech/LibriTTS), so feeding it other-language text
  won't produce correct pronunciation in the cloned voice regardless of
  phonemizer support. Kokoro/Piper don't clone your voice in other languages
  either — they're fixed synthetic voices per language. Multi-language voice
  cloning would need a different underlying model (e.g. XTTS-v2).
- `STYLETTS2_CHUNK_CHARS` in `voiceclone/core.py` controls how StyleTTS2's
  long-text splitter chunks input before phonemizing. The library's own
  default (420 raw characters) can still overflow the model's 512-token limit
  on text with heavy emphasis/punctuation; this package uses a tighter 250 by
  default.

## License

MIT — see [LICENSE](LICENSE). Note StyleTTS2 itself is also MIT-licensed
upstream; Kokoro-82M and Piper have their own separate licenses (Apache-2.0
and MIT respectively, as of this writing — verify before relying on this).
