#!/usr/bin/env python3
"""
CLI entry point: synthesize a WAV voiceover from a text file.

Usage:
    voiceclone --script build/script.txt --out build/voice.wav \
        --voice-ref-repo you/voice-reference-audio

    # env vars work too, so a workflow step can just set them:
    #   VOICE_REF_REPO=you/voice-reference-audio
    #   VOICE_REPO_PAT=<fine-grained read-only PAT>
    voiceclone --script build/script.txt --out build/voice.wav

    voiceclone --engine kokoro --voice am_fenrir
    voiceclone --engine piper --voice en_US-ryan-high
    voiceclone --engine espeak
"""
import argparse
import os
import sys
from pathlib import Path

from .core import VOICE_PATHS, AllEnginesFailedError, synthesize


def main():
    ap = argparse.ArgumentParser(
        prog="voiceclone",
        description="Synthesize a voiceover WAV, cloning a voice via StyleTTS2 "
                     "with Kokoro/Piper/espeak fallback.",
    )
    ap.add_argument("--script", default="build/script.txt", help="Text file to narrate.")
    ap.add_argument("--out", default="build/voice.wav")
    ap.add_argument("--engine", default="auto",
                     choices=["auto", "styletts2", "kokoro", "piper", "espeak", "openai"],
                     help="auto tries styletts2 -> kokoro -> piper -> espeak in order.")
    ap.add_argument("--voice", default=None,
                     help="Kokoro voice, Piper voice ID, or OpenAI voice name, "
                          "depending on --engine.")
    ap.add_argument("--voice-ref-repo", default=None,
                     help="owner/repo of a private GitHub repo holding the voice "
                          "reference sample (falls back to the VOICE_REF_REPO env "
                          "var). Required for --engine styletts2.")
    ap.add_argument("--voice-ref-file", default="reference_voice.mp3")
    ap.add_argument("--voice-repo-pat-env", default="VOICE_REPO_PAT",
                     help="Name of the env var holding a read-only GitHub PAT "
                          "scoped to --voice-ref-repo.")
    ap.add_argument("--voices-dir", default="voices")
    ap.add_argument("--openai-model", default="tts-1")
    args = ap.parse_args()

    text = Path(args.script).read_text(encoding="utf-8").strip()

    voice_ref_repo = args.voice_ref_repo or os.environ.get("VOICE_REF_REPO")
    kokoro_voice = args.voice or "am_fenrir"
    piper_voice = args.voice if args.voice in VOICE_PATHS else "en_US-amy-medium"
    openai_voice = args.voice or "onyx"

    try:
        used = synthesize(
            text,
            args.out,
            engine=args.engine,
            voice_ref_repo=voice_ref_repo,
            voice_ref_file=args.voice_ref_file,
            voice_repo_pat_env=args.voice_repo_pat_env,
            kokoro_voice=kokoro_voice,
            piper_voice=piper_voice,
            voices_dir=args.voices_dir,
            openai_model=args.openai_model,
            openai_voice=openai_voice,
        )
    except AllEnginesFailedError as exc:
        sys.exit(str(exc))

    print(f"[voiceclone] wrote {args.out} (engine={used})")


if __name__ == "__main__":
    main()
