#!/usr/bin/env bash
# Audio -> stems -> notes -> Strudel. Proof of concept for testing the quality on one song.
#
#   ./run.sh song.mp3
#   ./run.sh song.mp3 --start-bar 9 --bars 16     # only an excerpt in the Strudel code
#   ./run.sh song.mp3 --mode mix                  # without separation, for comparison
#   ./run.sh song.mp3 --six-stems                 # also guitar and piano
#   ./run.sh song.mp3 --simplify 0.8              # loops and parts: much shorter code
#   ./run.sh song.mp3 --as-audio vocals,other     # play vocals and the rest as audio
#
# All flags after the file name are passed on to pipeline.py (see: .venv/bin/python pipeline.py -h).
# Environment variables: CPU=1 installs the CPU build of PyTorch, REINSTALL=1 recreates the environment.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$root/.venv"
py="$venv/bin/python"

if [[ $# -lt 1 ]]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
audio="$1"; shift
[[ -f "$audio" ]] || { echo "File not found: $audio" >&2; exit 1; }
command -v uv >/dev/null || { echo "Missing uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "Missing ffmpeg (macOS: brew install ffmpeg, Debian/Ubuntu: sudo apt install ffmpeg)" >&2; exit 1; }

# --- Python environment (first run only) --------------------------------------
[[ "${REINSTALL:-0}" == 1 ]] && rm -rf "$venv"
if [[ ! -x "$py" ]]; then
    echo "First run: creating the Python environment. This takes a few minutes..."
    uv venv --python 3.12 "$venv"
    backend=auto; [[ "${CPU:-0}" == 1 ]] && backend=cpu
    if ! uv pip install --python "$py" --torch-backend "$backend" torch torchaudio 'audio-separator[cpu]' audioread muscriptor; then
        rm -rf "$venv"; echo "Package installation failed." >&2; exit 1
    fi
fi
"$py" -c "import torch; d = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else None); print('PyTorch', torch.__version__, '| GPU:', d or 'none (running on CPU, this takes longer)')"

# --- HuggingFace login (the MuScriptor model is free, but its license must be accepted) ---
if ! "$py" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)"; then
    echo
    echo "The MuScriptor model requires a free HuggingFace account:"
    echo "  1. Accept the license at https://huggingface.co/MuScriptor/muscriptor-medium"
    echo "  2. Create a read token at https://huggingface.co/settings/tokens"
    echo "  3. Paste the token below."
    "$py" -c "from huggingface_hub import login; login()"
fi

# --- Run the pipeline ---------------------------------------------------------
cd "$root"
"$py" pipeline.py "$(cd "$(dirname "$audio")" && pwd)/$(basename "$audio")" --out "$root/out" "$@"
