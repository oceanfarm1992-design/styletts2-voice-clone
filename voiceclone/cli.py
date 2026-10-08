#!/usr/bin/env python3
"""
CLI entry point: synthesize a WAV voiceover from a text file.

Usage:
    # env vars (as the GitHub Action sets them):
    #   VOICE_REF_REPO=you/voice-reference-audio
    #   VOICE_REPO_PAT=<fine-grained read-only PAT>
    #   HF_TOKEN=<Hugging Face read token, for Tamil (IndicF5)>
    voiceclone --script build/script.txt --out build/voice.wav
    voiceclone --language ta --style comedy --script build/script_ta.txt --out build/voice.wav

    voiceclone --engine kokoro --voice am_fenrir     # force one engine
    voiceclone --engine edge --language ta --rvc off # stock Tamil voice, no conversion
"""
import argparse
import os
import sys
from pathlib import Path

from .core import ALL_ENGINES, LANGUAGE_CHAINS, VOICE_PATHS, AllEnginesFailedError, synthesize


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="voiceclone",
        description="Synthesize a voiceover WAV in a cloned voice (English: StyleTTS2, "
                    "Tamil: IndicF5), converted with an RVC model of the same voice, "
                    "with stock-voice fallbacks.",
    )
    ap.add_argument("--script", default="build/script.txt", help="Text file to narrate.")
    ap.add_argument("--out", default="build/voice.wav")
    ap.add_argument("--language", default=os.environ.get("VOICE_LANGUAGE") or "en",
                    choices=sorted(LANGUAGE_CHAINS),
                    help="Language of the script (falls back to VOICE_LANGUAGE).")
    ap.add_argument("--style", default=os.environ.get("VOICE_STYLE") or "default",
                    help="Style block from the voice profile, e.g. comedy "
                         "(falls back to VOICE_STYLE).")
    ap.add_argument("--engine", default="auto", choices=["auto", *ALL_ENGINES],
                    help="auto walks the language's chain: "
                         + "; ".join(f"{k}: {' -> '.join(v)}" for k, v in LANGUAGE_CHAINS.items()))
    ap.add_argument("--rvc", default=os.environ.get("VOICE_RVC") or "auto",
                    choices=["auto", "on", "off"],
                    help="Convert the result with the profile's RVC model (auto = when "
                         "available, keeping unconverted audio if conversion fails).")
    ap.add_argument("--voice", default=None,
                    help="Kokoro voice, Piper voice ID, Edge voice or OpenAI voice name, "
                         "depending on --engine.")
    ap.add_argument("--voice-ref-repo", default=None,
                    help="owner/repo of the private voice reference repo "
                         "(falls back to VOICE_REF_REPO).")
    ap.add_argument("--voice-ref-file", default="reference_voice.mp3",
                    help="English timbre clip, used when the repo has no profile.json.")
    ap.add_argument("--voice-repo-pat-env", default="VOICE_REPO_PAT",
                    help="Name of the env var holding a read-only GitHub PAT "
                         "scoped to --voice-ref-repo.")
    ap.add_argument("--style-ref-file", default=os.environ.get("VOICE_STYLE_REF") or None,
                    help="Override the profile's English prosody clip "
                         "(falls back to VOICE_STYLE_REF).")
    ap.add_argument("--alpha", type=float, default=None,
                    help="StyleTTS2 timbre: 0 = exactly the reference voice.")
    ap.add_argument("--beta", type=float, default=None,
                    help="StyleTTS2 prosody: 0 = copy the style clip, 1 = invent from text.")
    ap.add_argument("--embedding-scale", type=float, default=None,
                    help="StyleTTS2 expressiveness; higher = more emotional.")
    ap.add_argument("--voices-dir", default="voices")
    ap.add_argument("--openai-model", default="tts-1")
    return ap


def main() -> None:
    args = _parser().parse_args()
    text = Path(args.script).read_text(encoding="utf-8").strip()
    if not text:
        sys.exit(f"{args.script} is empty.")

    try:
        used = synthesize(
            text,
            args.out,
            engine=args.engine,
            language=args.language,
            style=args.style,
            rvc=args.rvc,
            voice_ref_repo=args.voice_ref_repo,
            voice_ref_file=args.voice_ref_file,
            voice_repo_pat_env=args.voice_repo_pat_env,
            style_ref_file=args.style_ref_file,
            styletts2_alpha=args.alpha,
            styletts2_beta=args.beta,
            styletts2_embedding_scale=args.embedding_scale,
            kokoro_voice=args.voice or "am_fenrir",
            piper_voice=args.voice if args.voice in VOICE_PATHS else "en_US-amy-medium",
            edge_voice=args.voice if args.engine == "edge" else None,
            voices_dir=args.voices_dir,
            openai_model=args.openai_model,
            openai_voice=args.voice or "onyx",
        )
    except AllEnginesFailedError as exc:
        sys.exit(str(exc))

    print(f"[voiceclone] wrote {args.out} (engine={used})")


if __name__ == "__main__":
    main()
