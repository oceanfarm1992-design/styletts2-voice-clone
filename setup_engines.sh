#!/usr/bin/env bash
# Build one CPU-only virtualenv per engine under $VOICECLONE_HOME/venvs/<engine>,
# plus the Applio checkout and RVC inference weights. Linux (GitHub runners).
#
#   ./setup_engines.sh styletts2 indicf5 rvc kokoro
#
# Needs: python3 (3.10-3.12), git, curl. Installs uv if missing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${VOICECLONE_HOME:-$HOME/.voiceclone}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
APPLIO_REPO="https://github.com/IAHispano/Applio.git"
APPLIO_SHA="7b9f3fa0dde9f90946a5302b4ce4ab3410f12bb8"
APPLIO_WEIGHTS="https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

engines=("$@")
[ ${#engines[@]} -gt 0 ] || engines=(styletts2 indicf5 rvc)

command -v uv >/dev/null || "$PYTHON" -m pip install -q uv
mkdir -p "$HOME_DIR/venvs"

setup_venv() {
  local engine="$1" venv="$HOME_DIR/venvs/$1"
  echo "::group::voiceclone: $engine venv"
  # A venv restored from the Actions cache is reused; rebuild it if its
  # interpreter no longer runs (e.g. a different Python build).
  "$venv/bin/python" -c "" 2>/dev/null || { rm -rf "$venv"; uv venv -q -p "$PYTHON" "$venv"; }
  # unsafe-best-match: take torch from the CPU index, everything else from PyPI.
  uv pip install -q -p "$venv/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url "$TORCH_INDEX" -r "$HERE/requirements/$engine.txt"
  # Always refresh this package itself: a cached venv may hold an older copy
  # with the same version number.
  uv pip install -q -p "$venv/bin/python" --no-deps --reinstall-package styletts2-voice-clone "$HERE"
  echo "::endgroup::"
}

setup_applio() {
  local dir="$HOME_DIR/applio"
  echo "::group::voiceclone: Applio $APPLIO_SHA"
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$APPLIO_SHA" ]; then
    rm -rf "$dir" && mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" fetch -q --depth 1 "$APPLIO_REPO" "$APPLIO_SHA"
    git -C "$dir" checkout -q FETCH_HEAD
  fi
  local f
  for f in predictors/rmvpe.pt embedders/contentvec/pytorch_model.bin embedders/contentvec/config.json; do
    local dest="$dir/rvc/models/$f"
    [ -s "$dest" ] || curl -fsSL --retry 3 --create-dirs -o "$dest" "$APPLIO_WEIGHTS/$f"
  done
  echo "::endgroup::"
}

for engine in "${engines[@]}"; do
  case "$engine" in
    styletts2|indicf5|kokoro) setup_venv "$engine" ;;
    rvc) setup_venv rvc; setup_applio ;;
    *) echo "unknown engine: $engine" >&2; exit 2 ;;
  esac
done

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "VOICECLONE_HOME=$HOME_DIR" >> "$GITHUB_ENV"
  echo "APPLIO_DIR=$HOME_DIR/applio" >> "$GITHUB_ENV"
fi
echo "voiceclone engines ready in $HOME_DIR: ${engines[*]}"
