#!/usr/bin/env bash
# Build the training environment on the NSUT DGX, from a clean container.
#
# Every line below exists because something broke without it. Read the comments
# before changing any pin: none of them are cosmetic.

set -euo pipefail

VENV=/workspace/venv

# 1. The NGC image lists pypi.ngc.nvidia.com as an extra index in four pip.conf
#    files (/etc/pip.conf, /etc/xdg/pip/pip.conf, /root/.pip/pip.conf,
#    /root/.config/pip/pip.conf) and that host does not resolve from inside the
#    container. Every install then retries five times and dies looking exactly
#    like a dead network, while curl to PyPI returns 200 throughout.
#    PIP_CONFIG_FILE=/dev/null suppresses the global config but NOT the
#    site-level one, so the index is also overridden on the command line.
export PIP_CONFIG_FILE=/dev/null
PIPI=(--index-url https://pypi.org/simple --extra-index-url https://pypi.org/simple)

# 2. pip 24.3.1, which the image ships, crashes in its own version parser:
#    TypeError: expected string or bytes-like object, got 'NoneType'
#    at pip/_vendor/packaging/version.py line 200. Upgrading fixes it.
python3 -m pip install -q --upgrade pip "${PIPI[@]}"

# 3. Training runs in its own venv, NOT the system Python. Three constraints
#    are mutually exclusive in one interpreter:
#      - coqui-tts needs NumPy 2 (via librosa >= 0.11)
#      - NVIDIA's torch 2.6.0a0+nv24.12 is compiled against NumPy 1, and with
#        NumPy 2 it imports but its NumPy bridge dies silently
#      - coqui-tts needs torchaudio, and PyPI builds torchaudio against
#        UPSTREAM torch; against the NGC build it fails to load with
#        undefined symbol: _ZNK5torch8autograd4Node4nameEv
#    So the venv takes a matched upstream pair and the container's system
#    Python is left alone, which keeps the already-run data pipeline working.
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip "${PIPI[@]}"
"$VENV/bin/python" -m pip install -q torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

# 4. The codec extra is required: from PyTorch 2.9 audio IO moved to
#    torchcodec, and coqui-tts raises TORCHCODEC_IMPORT_ERROR without it.
# 5. transformers must stay on 4.x. transformers 5 removed
#    transformers.pytorch_utils.isin_mps_friendly, which coqui still imports
#    through its tortoise layer, so the whole package fails to import.
"$VENV/bin/python" -m pip install -q 'coqui-tts[codec]' 'transformers<5' "${PIPI[@]}"

"$VENV/bin/python" - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import torch, numpy, librosa, transformers
from TTS.tts.models.vits import Vits
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
print("torch       ", torch.__version__)
print("numpy       ", numpy.__version__)
print("librosa     ", librosa.__version__)
print("transformers", transformers.__version__)
print("gpu         ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("bf16        ", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
print("models      ", Vits.__name__, ForwardTTS.__name__, "importable")
PY
echo "environment ready at $VENV"
