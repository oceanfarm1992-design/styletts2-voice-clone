# styletts2-voice-clone

One voice engine for all my video pipelines. It narrates **English and Tamil** in
my own cloned voice, including a **comedy** speaking style. It runs **CPU-only
inside GitHub Actions**. The only GPU step is training the voice model, done
once on a local machine.

```
                 English                          Tamil
base speech      StyleTTS2 (clone)                IndicF5 (clone, copies tone)
  fallbacks      Kokoro -> Piper -> Edge -> espeak Edge (ta-LK) -> espeak
                          \__________________ _________________/
                                             v
voice conversion        RVC model trained on my voice (local GPU)
                                             v
                                         voice.wav
```

Each stage falls back on failure. A missing secret or a flaky download still
produces audio; it just uses a stock voice. RVC converts every base engine,
fallbacks included, so even a stock voice comes out sounding like me.

This repo is public and holds **code only**. Voice clips, the trained RVC model
and `profile.json` live in a separate **private** repo (`voice-reference-audio`).
The engine fetches them at runtime with a read-only token. Never commit voice
data here.

## Use it from another repo's workflow

```yaml
- uses: actions/checkout@v4
- name: Narrate
  id: voice
  uses: oceanfarm1992-design/styletts2-voice-clone@main
  with:
    script: build/script.txt          # omit to only install
    out: build/voice.wav
    language: ta                      # en | ta
    style: comedy                     # a style block from profile.json
    voice-ref-repo: ${{ vars.VOICE_REF_REPO }}
    voice-repo-pat: ${{ secrets.VOICE_REPO_PAT }}
- run: echo "made with ${{ steps.voice.outputs.engine }}"   # e.g. indicf5+rvc
```

After the action runs, later steps can also call `voiceclone ...` or use
`from voiceclone import synthesize` from the job's Python. The secrets have to
be in the environment of that step:

```python
from voiceclone import synthesize

used = synthesize("வணக்கம்!", "build/voice.wav", language="ta", style="comedy")
```

Heavy engines stay loaded between calls, so narrating twenty scenes in one job
loads each model once.

Inputs: `engines` (default `styletts2 indicf5 rvc`; add `kokoro` for that
fallback), `python-version`, plus the ones above. Build only what the job needs.
For example, an English-only pipeline can use `engines: styletts2 rvc`.

### Faster: narrate on several runners in parallel

Tamil (IndicF5) is slow on CPU: about 3.5 minutes per sentence on a 4-core
runner. The reusable `narrate` workflow splits the script at sentence
boundaries and narrates each part on its own runner. Then it joins the parts
and uploads the result as one artifact. Every sentence is generated exactly as
in a single job, so quality is identical. Wall time drops roughly by `parts`.
Each runner adds about 3-5 minutes of setup.

```yaml
jobs:
  voice:
    uses: oceanfarm1992-design/styletts2-voice-clone/.github/workflows/narrate.yml@main
    with:
      text: ${{ needs.script.outputs.text }}
      language: ta
      style: comedy
      parts: 6                        # public repos get up to 20 parallel jobs
      artifact-name: voiceover
    secrets: inherit                  # VOICE_REPO_PAT
  render:
    needs: voice
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: {name: voiceover}       # -> voice.wav
```

### Secrets and variables per consuming repo

| Name | Kind | What |
|---|---|---|
| `VOICE_REF_REPO` | variable | `oceanfarm1992-design/voice-reference-audio` |
| `VOICE_REPO_PAT` | secret | Fine-grained PAT, **read-only** on that one private repo |
| `HF_TOKEN` | secret | *Optional fallback.* The action installs IndicF5 from the private repo's `indicf5-v1` release (made once by its `mirror-indicf5` workflow), so the voice PAT is enough. Needed only if that release is missing. |

## The private voice repo

```
voice-reference-audio/
  profile.json
  reference_voice.mp3            English timbre reference
  refs/comedy_en_style.wav       English comedy prosody clip
  refs/ta_comedy.wav             Tamil reference (8-12 s, exact transcript in profile.json)
  rvc/my_voice.pth               RVC model (from training, below)
  rvc/my_voice.index
```

```json
{
  "rvc": {"model": "rvc/my_voice.pth", "index": "rvc/my_voice.index",
          "pitch": 0, "index_rate": 0.75, "protect": 0.5},
  "styles": {
    "default": {
      "en": {"voice": "reference_voice.mp3"},
      "ta": {"ref_audio": "refs/ta_comedy.wav", "ref_text": "<exact words spoken in the clip>"}
    },
    "comedy": {
      "en": {"voice": "reference_voice.mp3", "style": "refs/comedy_en_style.wav",
             "beta": 0.2, "embedding_scale": 1.5},
      "ta": {"ref_audio": "refs/ta_comedy.wav", "ref_text": "..."}
    }
  }
}
```

A style without a block for a language falls back to `default`. IndicF5 copies
the **tone of its reference clip**: a comedy clip gives comedy delivery. To add
a new style, add a clip and a style block. You don't need to change any code.

A repo without `profile.json` keeps working the 0.2 way: English only, using
`reference_voice.mp3`, with no RVC.

## Training the RVC model (local GPU, once)

Done with [Applio](https://github.com/IAHispano/Applio) on a local NVIDIA GPU.
3-10 minutes of clean speech is plenty:

1. Strip background music. For example, run `audio-separator` with
   `model_bs_roformer_ep_317_sdr_12.9755.ckpt`, keeping the Vocals stem.
2. In Applio, run `preprocess` → `extract` (rmvpe, contentvec) → `train`
   (40 kHz, HiFi-GAN, ~250 epochs) → `index`.
3. Copy `logs/<name>/<name>_<epoch>e_<step>s.pth` and `logs/<name>/<name>.index`
   into the private repo's `rvc/` folder, and update `profile.json`.

Inference in CI uses the same Applio commit (`APPLIO_SHA` in
`setup_engines.sh`) on CPU.

## CLI

```bash
voiceclone --script build/script.txt --out build/voice.wav                 # English, default style
voiceclone --language ta --style comedy --script s.txt --out v.wav         # Tamil comedy
voiceclone --engine edge --language ta --rvc off --script s.txt            # stock Tamil voice only
```

`--rvc auto` (default) converts when the profile has a model, and keeps the
unconverted audio if conversion fails. `on` treats a conversion failure as
that engine failing. `off` skips conversion. All options: `voiceclone --help`.

## How it runs

- `setup_engines.sh` builds one CPU-only virtualenv per engine under
  `~/.voiceclone/venvs/` from `requirements/<engine>.txt`. StyleTTS2, IndicF5
  and Applio pin mutually incompatible `transformers`/`numpy`/`huggingface-hub`
  versions, so they can't share one environment.
- `voiceclone/pool.py` runs each engine as a long-lived worker
  (`python -m voiceclone.workers.<engine>`) and sends it JSON lines.
- If an engine has no venv, its worker runs on the current interpreter. So the
  0.2 setup still works:
  `pip install "styletts2-voice-clone[styletts2] @ git+..."`.

## Workflow in this repo

`.github/workflows/voice.yml` runs the unit tests on every push. Run it manually
(**Actions → voice → Run workflow**) to narrate a sample in GitHub Actions and
download the WAV. A weekly scheduled run narrates English and Tamil and fails
if the cloned voice wasn't used.

## Notes

- Measured on 4 CPU threads, with models loaded, for one 8.6 s sentence:
  StyleTTS2 ~30 s, RVC 17 s, and IndicF5 (16 steps, 6 s reference clip) 199 s.
  The 11 s clip took 324 s and 32 steps took twice as long, both with no
  better quality. 8 steps and int8 quantization were faster but slurred
  words. Keep the Tamil reference clip at 5-6 s.
- StyleTTS2 is English-only and IndicF5 covers 11 Indian languages. Adding
  another Indian language is a new entry in `LANGUAGE_CHAINS` plus a reference
  clip.
- Licenses: StyleTTS2 MIT, IndicF5 MIT (gated download), Applio MIT,
  Kokoro Apache-2.0, Piper MIT. Verify before relying on this.

## License

MIT. See [LICENSE](LICENSE).
