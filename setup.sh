#!/usr/bin/env bash
# setup.sh — rebuild the two venvs on a fresh machine (they're gitignored, so a
# clone won't have them). After this + placing your token, run_pipeline.sh works.
#
#   git clone https://github.com/CharlieK123/RocketJEPA.git && cd RocketJEPA
#   ./setup.sh
#   printf '%s' 'YOUR_GC_TOKEN' > .ballchasing_token
#   ./run_pipeline.sh /path/to/data 94000
set -euo pipefail
cd "$(dirname "$0")"

# uv gives a clean standalone CPython 3.11 for carball (Homebrew's 3.11 is broken
# for it — pyexpat/libexpat mismatch). Install uv if it isn't already present.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "==> decode venv (.venv-decode): python 3.11 + carball + zstandard"
uv venv --python 3.11 .venv-decode
uv pip install --python .venv-decode/bin/python sprocket-carball zstandard

echo "==> download venv (.venv): requests"
uv venv .venv
uv pip install --python .venv/bin/python requests
# If you also train / run the loader on this machine, add: torch numpy zstandard

echo "==> verifying"
.venv-decode/bin/python -c "import carball, zstandard; print('   decode venv OK')"
.venv/bin/python -c "import requests; print('   download venv OK')"

cat <<'NEXT'

setup complete. next:
  1) put your GC token in .ballchasing_token   (never commit it — it's gitignored)
  2) ./run_pipeline.sh /path/to/data 94000      (data dir on a volume with space)
NEXT
