"""Voice profile: which reference clips and RVC model to use, per style and language.

The profile and every file it names live in a separate *private* repo (never in
this public one) and are fetched at runtime with a read-only token, or read from
a local folder (VOICE_REF_DIR) for testing. Layout of that repo:

    profile.json
    reference_voice.mp3          # English timbre reference
    rvc/my_voice.pth             # RVC model trained on the GPU (see README)
    rvc/my_voice.index
    refs/...                     # style / Tamil reference clips

profile.json:
    {
      "rvc": {"model": "rvc/my_voice.pth", "index": "rvc/my_voice.index",
              "pitch": 0, "index_rate": 0.75, "protect": 0.5},
      "styles": {
        "default": {"en": {"voice": "reference_voice.mp3"},
                    "ta": {"ref_audio": "refs/ta.wav", "ref_text": "<exact transcript>"}},
        "comedy":  {"en": {"voice": "reference_voice.mp3", "style": "refs/comedy.wav",
                           "beta": 0.2, "embedding_scale": 1.5},
                    "ta": {"ref_audio": "refs/ta_comedy.wav", "ref_text": "..."}}
      }
    }

A repo without profile.json still works the pre-0.3 way: English only, timbre
from reference_voice.mp3 (or voice_ref_file), no RVC.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

PROFILE_FILE = "profile.json"


class ProfileError(RuntimeError):
    """The voice reference source is missing, unreachable, or misconfigured."""


@dataclass(frozen=True)
class RvcSettings:
    model: str
    index: str
    pitch: int = 0
    index_rate: float = 0.75
    protect: float = 0.5


@dataclass(frozen=True)
class Profile:
    styles: dict = field(default_factory=dict)  # style -> language -> settings
    rvc: RvcSettings | None = None

    def style_for(self, style: str, language: str) -> dict:
        """Settings for (style, language), falling back to the default style."""
        for name in (style, "default"):
            settings = self.styles.get(name, {}).get(language)
            if settings:
                return dict(settings)
        return {}


def parse_profile(data: dict) -> Profile:
    rvc = data.get("rvc")
    if rvc is not None:
        if not (rvc.get("model") and rvc.get("index")):
            raise ProfileError("profile.json 'rvc' needs both 'model' and 'index'.")
        rvc = RvcSettings(model=rvc["model"], index=rvc["index"],
                          pitch=int(rvc.get("pitch", 0)),
                          index_rate=float(rvc.get("index_rate", 0.75)),
                          protect=float(rvc.get("protect", 0.5)))
    styles = data.get("styles", {})
    if not isinstance(styles, dict):
        raise ProfileError("profile.json 'styles' must be an object.")
    return Profile(styles=styles, rvc=rvc)


def legacy_profile(voice_ref_file: str, style_ref_file: str | None) -> Profile:
    en = {"voice": voice_ref_file}
    if style_ref_file:
        en["style"] = style_ref_file
    return Profile(styles={"default": {"en": en}})


class VoiceSource:
    """Fetches files from the private reference repo (or a local folder),
    caching each one for the life of the process."""

    def __init__(self, repo: str | None = None, pat: str | None = None,
                 local_dir: str | None = None, cache_dir: str | None = None):
        self.repo = repo
        self.pat = pat
        self.local_dir = Path(local_dir) if local_dir else None
        self.cache_dir = Path(cache_dir or "build/.voice_refs")
        self._paths: dict[str, str] = {}

    @classmethod
    def from_env(cls, repo: str | None, pat_env: str, cache_dir: str | None) -> "VoiceSource":
        return cls(repo=repo or os.environ.get("VOICE_REF_REPO") or None,
                   pat=os.environ.get(pat_env, "").strip() or None,
                   local_dir=os.environ.get("VOICE_REF_DIR") or None,
                   cache_dir=cache_dir)

    def fetch(self, name: str) -> str:
        """Local path to `name` from the reference source. Raises ProfileError
        if it can't be found; a missing optional file is the caller's call."""
        if name in self._paths:
            return self._paths[name]
        if self.local_dir:
            path = self.local_dir / name
            if not path.is_file():
                raise ProfileError(f"{name} not found in VOICE_REF_DIR {self.local_dir}")
            self._paths[name] = str(path.resolve())
            return self._paths[name]
        if not self.repo:
            raise ProfileError("No voice reference repo configured "
                               "(pass voice_ref_repo= or set VOICE_REF_REPO).")
        if not self.pat:
            raise ProfileError("No PAT available to fetch the voice reference files "
                               "(set the env var named by voice_repo_pat_env).")
        dest = self.cache_dir / name
        url = f"https://api.github.com/repos/{self.repo}/contents/{name}"
        headers = {"Authorization": f"Bearer {self.pat}",
                   "Accept": "application/vnd.github.raw+json"}
        try:
            with requests.get(url, headers=headers, timeout=300, stream=True) as resp:
                if resp.status_code == 404:
                    raise ProfileError(f"{name} not found in {self.repo}")
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except requests.RequestException as exc:
            raise ProfileError(f"Could not fetch {name} from {self.repo}: {exc}") from exc
        self._paths[name] = str(dest.resolve())
        return self._paths[name]

    def profile(self, voice_ref_file: str, style_ref_file: str | None) -> Profile:
        try:
            path = self.fetch(PROFILE_FILE)
        except ProfileError as exc:
            if "not found" not in str(exc):
                raise
            return legacy_profile(voice_ref_file, style_ref_file)
        with open(path, encoding="utf-8") as f:
            return parse_profile(json.load(f))
