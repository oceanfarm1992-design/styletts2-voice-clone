"""Split a script across parallel CI jobs and join their audio back together.

Used by the reusable `narrate` workflow: one job splits the script into
sentence-aligned parts, a matrix of jobs narrates one part each (so a slow
engine like IndicF5 runs on several runners at once), and a last job joins the
WAVs. Every sentence is synthesized exactly as it would be in one job.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")
JOIN_SAMPLE_RATE = 44100
JOIN_GAP_SECONDS = 0.3


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def split_script(text: str, parts: int) -> list[str]:
    """Split into at most `parts` contiguous, sentence-aligned pieces of
    roughly equal length (in characters, which tracks speaking time)."""
    sentences = split_sentences(text)
    if parts < 1:
        raise ValueError("parts must be >= 1")
    if not sentences:
        return []
    n = len(sentences)
    parts = min(parts, n)
    target = sum(len(s) for s in sentences) / parts
    pieces: list[list[str]] = [[] for _ in range(parts)]
    done = 0  # characters before the current sentence
    index = -1  # piece of the previous sentence
    for i, sentence in enumerate(sentences):
        # The piece whose share of the text contains this sentence's midpoint...
        ideal = int((done + len(sentence) / 2) // target)
        # ...but never skip a piece (no empty parts) and always leave at least
        # one sentence for every piece still to come.
        index = max(min(ideal, index + 1, parts - 1), parts - (n - i), index)
        pieces[index].append(sentence)
        done += len(sentence)
    return [" ".join(p) for p in pieces]


def join_wavs(paths: list[str], out: str, gap: float = JOIN_GAP_SECONDS) -> None:
    """Concatenate WAVs in order with a short pause between them, resampled to
    one rate and mono (parts made by different fallback engines can differ)."""
    if not paths:
        raise ValueError("Nothing to join.")
    inputs: list[str] = []
    for p in paths:
        inputs += ["-i", p]
    chains = []
    labels = []
    for i in range(len(paths)):
        pad = f",apad=pad_dur={gap}" if i < len(paths) - 1 else ""
        chains.append(f"[{i}:a]aresample={JOIN_SAMPLE_RATE},aformat=channel_layouts=mono{pad}[a{i}]")
        labels.append(f"[a{i}]")
    graph = ";".join(chains) + f";{''.join(labels)}concat=n={len(paths)}:v=0:a=1[out]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
           "-map", "[out]", out]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg join failed: {proc.stderr.decode('utf-8', 'replace')}")


def split_main() -> None:
    ap = argparse.ArgumentParser(prog="voiceclone-split",
                                 description="Split a script into sentence-aligned parts.")
    ap.add_argument("--script", required=True)
    ap.add_argument("--parts", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    pieces = split_script(Path(args.script).read_text(encoding="utf-8"), args.parts)
    if not pieces:
        sys.exit(f"{args.script} has no text.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, piece in enumerate(pieces):
        name = f"part_{i:02d}"
        (out_dir / f"{name}.txt").write_text(piece + "\n", encoding="utf-8")
        names.append(name)
    print(json.dumps(names))


def join_main() -> None:
    ap = argparse.ArgumentParser(prog="voiceclone-join",
                                 description="Join part WAVs in order into one file.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=JOIN_GAP_SECONDS)
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()
    join_wavs(args.wavs, args.out, args.gap)
    print(f"[voiceclone] joined {len(args.wavs)} parts into {args.out}")
