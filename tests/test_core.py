import json

import pytest

from voiceclone import core
from voiceclone.profile import Profile, ProfileError, RvcSettings, VoiceSource, parse_profile

PROFILE = {
    "rvc": {"model": "rvc/m.pth", "index": "rvc/m.index", "pitch": 2},
    "styles": {
        "default": {"en": {"voice": "ref.mp3"}, "ta": {"ref_audio": "ta.wav", "ref_text": "t"}},
        "comedy": {"en": {"voice": "ref.mp3", "style": "funny.wav", "beta": 0.2}},
    },
}


@pytest.fixture
def ref_dir(tmp_path, monkeypatch):
    (tmp_path / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
    monkeypatch.setenv("VOICE_REF_DIR", str(tmp_path))
    monkeypatch.setattr(core, "_SOURCES", {})
    return tmp_path


@pytest.fixture
def calls(monkeypatch):
    """Record engine calls; engines named in `failing` raise, others write audio."""
    log = {"base": [], "rvc": [], "failing": set(), "rvc_fails": False}

    def fake_base(eng, text, out_wav, language, source, settings, *rest):
        log["base"].append((eng, language, dict(settings)))
        if eng in log["failing"]:
            raise RuntimeError(f"{eng} broke")
        with open(out_wav, "wb") as f:
            f.write(b"RIFF base")

    def fake_rvc(in_wav, out_wav, source, profile):
        log["rvc"].append(profile.rvc)
        if log["rvc_fails"]:
            raise RuntimeError("rvc broke")
        with open(out_wav, "wb") as f:
            f.write(b"RIFF converted")

    monkeypatch.setattr(core, "_run_base", fake_base)
    monkeypatch.setattr(core, "run_rvc", fake_rvc)
    return log


def test_english_uses_styletts2_then_rvc(ref_dir, calls, tmp_path):
    out = tmp_path / "out" / "v.wav"
    used = core.synthesize("hi", str(out))
    assert used == "styletts2+rvc"
    assert out.read_bytes() == b"RIFF converted"
    assert calls["rvc"] == [RvcSettings("rvc/m.pth", "rvc/m.index", pitch=2)]


def test_tamil_falls_back_to_edge_when_indicf5_fails(ref_dir, calls, tmp_path):
    calls["failing"].add("indicf5")
    used = core.synthesize("வணக்கம்", str(tmp_path / "v.wav"), language="ta")
    assert used == "edge+rvc"
    assert [c[0] for c in calls["base"]] == ["indicf5", "edge"]


def test_comedy_style_and_caller_overrides(ref_dir, calls, tmp_path):
    core.synthesize("hi", str(tmp_path / "v.wav"), style="comedy", styletts2_embedding_scale=1.7)
    settings = calls["base"][0][2]
    assert settings == {"voice": "ref.mp3", "style": "funny.wav", "beta": 0.2, "embedding_scale": 1.7}


def test_unknown_style_falls_back_to_default(ref_dir, calls, tmp_path):
    core.synthesize("வணக்கம்", str(tmp_path / "v.wav"), language="ta", style="comedy")
    assert calls["base"][0][2] == {"ref_audio": "ta.wav", "ref_text": "t"}


def test_rvc_failure_keeps_unconverted_audio(ref_dir, calls, tmp_path):
    calls["rvc_fails"] = True
    out = tmp_path / "v.wav"
    assert core.synthesize("hi", str(out)) == "styletts2"
    assert out.read_bytes() == b"RIFF base"


def test_strict_rvc_failure_moves_to_next_engine(ref_dir, calls, tmp_path):
    calls["rvc_fails"] = True
    with pytest.raises(core.AllEnginesFailedError):
        core.synthesize("hi", str(tmp_path / "v.wav"), rvc="on")
    assert len(calls["base"]) == len(core.LANGUAGE_CHAINS["en"])


def test_rvc_off_writes_base_audio_directly(ref_dir, calls, tmp_path):
    out = tmp_path / "v.wav"
    assert core.synthesize("hi", str(out), rvc="off") == "styletts2"
    assert out.read_bytes() == b"RIFF base" and calls["rvc"] == []


def test_all_engines_failing_raises(ref_dir, calls, tmp_path):
    calls["failing"].update(core.LANGUAGE_CHAINS["ta"])
    with pytest.raises(core.AllEnginesFailedError, match="espeak broke"):
        core.synthesize("x", str(tmp_path / "v.wav"), language="ta")


def test_missing_profile_source_still_allows_stock_voices(monkeypatch, calls, tmp_path):
    monkeypatch.delenv("VOICE_REF_DIR", raising=False)
    monkeypatch.delenv("VOICE_REF_REPO", raising=False)
    monkeypatch.setattr(core, "_SOURCES", {})
    used = core.synthesize("hi", str(tmp_path / "v.wav"), engine="edge")
    assert used == "edge" and calls["rvc"] == []


def test_unsupported_language_rejected(tmp_path):
    with pytest.raises(ValueError):
        core.synthesize("x", str(tmp_path / "v.wav"), language="fr")


def test_repo_without_profile_uses_legacy_english_settings(tmp_path):
    source = VoiceSource(local_dir=str(tmp_path))
    profile = source.profile("reference_voice.mp3", "style.wav")
    assert profile == Profile(styles={"default": {"en": {"voice": "reference_voice.mp3",
                                                          "style": "style.wav"}}})


def test_parse_profile_rejects_incomplete_rvc():
    with pytest.raises(ProfileError):
        parse_profile({"rvc": {"model": "m.pth"}})


def test_local_fetch_of_missing_file_raises(tmp_path):
    with pytest.raises(ProfileError, match="not found"):
        VoiceSource(local_dir=str(tmp_path)).fetch("nope.wav")

