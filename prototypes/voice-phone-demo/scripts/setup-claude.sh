#!/usr/bin/env bash
# Setup for --brain claude: Python venv + only the speech models (~1.5 GB).
# The brain is Claude through the local Claude Code login (subscription), so the
# 20 GB Qwen download in setup.sh is skipped. Uses uv for a pinned Python 3.12.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -d .venv ]] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -q -r requirements.txt

.venv/bin/hf download mlx-community/silero-vad
.venv/bin/hf download mlx-community/smart-turn-v3
.venv/bin/hf download mlx-community/nemotron-3.5-asr-streaming-0.6b
.venv/bin/hf download mlx-community/parakeet-tdt-0.6b-v3
.venv/bin/hf download gradium/phonon-31m-en --exclude "*.gguf"
for id in NbpkqMVS3CJeq2j8 4SZHfMpw-p46Ywgs vjFyp2BcCyTlUmYb NIWIOv5vTi5eRMj8 bXwrkHhURL1VFJdG \
          nGzMHMPAWGaJFHOo LMDIYE0Oa377Qh6Z 40QrMVgAxFiUnP9f d9XCyhjo4JhkHF1d; do
  .venv/bin/hf download gradium/phonon-voices "voices/$id.safetensors" >/dev/null
done
(cd ../../workspace/rose-agent/server/claude && bun install --frozen-lockfile 2>/dev/null || bun install)
echo "Setup done. Start Rose with: bash scripts/rose.sh --brain claude"
