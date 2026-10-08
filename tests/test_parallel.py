import shutil
import subprocess

import pytest

from voiceclone.parallel import join_wavs, split_script, split_sentences

TEXT = "ஒன்று. இரண்டு! மூன்று?\nநான்கு। Five is here. Six."


def test_split_sentences_handles_tamil_and_english_endings():
    assert split_sentences(TEXT) == ["ஒன்று.", "இரண்டு!", "மூன்று?", "நான்கு।",
                                     "Five is here.", "Six."]


def test_split_script_keeps_order_and_every_sentence():
    pieces = split_script(TEXT, 3)
    assert len(pieces) == 3
    assert " ".join(pieces).split() == " ".join(split_sentences(TEXT)).split()


def test_split_script_never_returns_empty_parts():
    assert split_script("One. Two.", 5) == ["One.", "Two."]
    long_then_short = "A" * 200 + ". B. C. D."
    assert all(split_script(long_then_short, 4))
    assert len(split_script(long_then_short, 4)) == 4


def test_split_script_balances_length():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    sizes = [len(p) for p in split_script(text, 4)]
    assert max(sizes) - min(sizes) <= 60


def test_split_script_empty_and_invalid():
    assert split_script("   ", 3) == []
    with pytest.raises(ValueError):
        split_script("x.", 0)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_join_wavs_concatenates_with_gaps(tmp_path):
    paths = []
    for i, rate in enumerate((24000, 40000)):
        p = tmp_path / f"p{i}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"sine=frequency=440:duration=1:sample_rate={rate}", str(p)], check=True)
        paths.append(str(p))
    out = tmp_path / "joined.wav"
    join_wavs(paths, str(out), gap=0.5)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert abs(float(probe.stdout) - 2.5) < 0.05
