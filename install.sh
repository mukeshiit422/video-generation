#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Video Generator Agent – Full Installation Script
# Supports: macOS (Apple Silicon / Intel), Linux (CUDA)
# Uses only free, open-source models and tools
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PYTHON="python3.11"
VENV_DIR="venv"
THIRD_PARTY="third_party"
MODELS_DIR="models"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Video Generator Agent – Installer      ║${NC}"
echo -e "${CYAN}║   Open-Source • Local • Private          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 0. Prerequisites ──────────────────────────────────────────────────────────
info "Checking prerequisites..."

command -v git >/dev/null 2>&1 || error "git is required. Install via: brew install git"

# Install system deps (macOS)
if [[ "$(uname)" == "Darwin" ]]; then
    command -v brew >/dev/null 2>&1 || error "Homebrew required: https://brew.sh"
    info "Installing system dependencies via Homebrew..."
    brew install ffmpeg cmake pkg-config 2>/dev/null || warn "Some brew packages may already be installed"
    brew install boost 2>/dev/null || true
    # Python 3.11 required – Coqui TTS does not support 3.12+
    if ! command -v python3.11 >/dev/null 2>&1; then
        info "Installing Python 3.11 (required for Coqui TTS)..."
        brew install python@3.11
    fi
fi

command -v $PYTHON >/dev/null 2>&1 || error "python3.11 not found. Run: brew install python@3.11"

ok "Prerequisites satisfied"

# ── 1. Virtual Environment ────────────────────────────────────────────────────
info "Creating Python 3.11 virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q
ok "Virtual environment ready"

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
info "Installing PyTorch..."
if [[ "$(uname)" == "Darwin" ]]; then
    # Apple Silicon / Intel Mac – MPS support built in
    pip install torch torchvision torchaudio -q
else
    # Linux with CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
fi
ok "PyTorch installed"

# ── 3. Python Requirements ────────────────────────────────────────────────────
info "Installing Python requirements..."
pip install -r requirements.txt -q
ok "Python requirements installed"

# ── 3b. Patch basicsr – functional_tensor removed in torchvision 0.16+ ───────
info "Patching basicsr for torchvision compatibility..."
BASICSR_FILE="$VENV_DIR/lib/python3.11/site-packages/basicsr/data/degradations.py"
if [ -f "$BASICSR_FILE" ]; then
    sed -i '' 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$BASICSR_FILE"
    ok "basicsr patched"
else
    warn "basicsr degradations.py not found – patch skipped"
fi

# ── 4. CLIP (required by SadTalker) ──────────────────────────────────────────
info "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git -q 2>/dev/null || \
    pip install openai-clip -q || warn "CLIP install failed – will fall back"
ok "CLIP installed"

# ── 5. Third-party Repos ──────────────────────────────────────────────────────
mkdir -p "$THIRD_PARTY"

# SadTalker – best single-image talking-head generator
info "Cloning SadTalker..."
if [ ! -d "$THIRD_PARTY/SadTalker" ]; then
    git clone https://github.com/OpenTalker/SadTalker.git "$THIRD_PARTY/SadTalker" --depth 1
    ok "SadTalker cloned"
else
    ok "SadTalker already present (skipping clone)"
fi

# Install SadTalker requirements
info "Installing SadTalker dependencies..."
pip install -r "$THIRD_PARTY/SadTalker/requirements.txt" -q 2>/dev/null || \
    warn "Some SadTalker deps may need manual install"

# Wav2Lip – alternative lip-sync engine
info "Cloning Wav2Lip..."
if [ ! -d "$THIRD_PARTY/Wav2Lip" ]; then
    git clone https://github.com/Rudrabha/Wav2Lip.git "$THIRD_PARTY/Wav2Lip" --depth 1
    ok "Wav2Lip cloned"
else
    ok "Wav2Lip already present"
fi

# Real-ESRGAN – video upscaling
info "Cloning Real-ESRGAN..."
if [ ! -d "$THIRD_PARTY/Real-ESRGAN" ]; then
    git clone https://github.com/xinntao/Real-ESRGAN.git "$THIRD_PARTY/Real-ESRGAN" --depth 1
    cd "$THIRD_PARTY/Real-ESRGAN" && pip install -e . -q && cd ../..
    ok "Real-ESRGAN installed"
else
    ok "Real-ESRGAN already present"
fi

# GFPGAN – face restoration
info "Cloning GFPGAN..."
if [ ! -d "$THIRD_PARTY/GFPGAN" ]; then
    git clone https://github.com/TencentARC/GFPGAN.git "$THIRD_PARTY/GFPGAN" --depth 1
    cd "$THIRD_PARTY/GFPGAN" && pip install -e . -q && cd ../..
    ok "GFPGAN installed"
else
    ok "GFPGAN already present"
fi

# ── 6. Create Directory Structure ─────────────────────────────────────────────
info "Creating project directories..."
mkdir -p \
    "$MODELS_DIR/sadtalker/checkpoints" \
    "$MODELS_DIR/sadtalker/gfpgan/weights" \
    "$MODELS_DIR/realesrgan" \
    "$MODELS_DIR/gfpgan" \
    "$MODELS_DIR/wav2lip" \
    "$MODELS_DIR/tts" \
    "outputs/temp" \
    "characters" \
    "scripts" \
    "training_data/metadata" \
    "training_data/videos" \
    "training_data/audio" \
    "checkpoints" \
    "logs"
ok "Directories created"

# ── 6b. Patch numpy 2.x / torchvision incompatibilities ──────────────────────
info "Patching dependency incompatibilities..."
python3 patch_dependencies.py
ok "Patches applied"

# ── 7. Download Models ────────────────────────────────────────────────────────
info "Downloading pre-trained models (this may take a while)..."
source "$VENV_DIR/bin/activate"
python3 setup_models.py --all 2>&1 | tail -20 || warn "Model download incomplete – run: python setup_models.py --all"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Installation Complete!                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Activate env:  source venv/bin/activate"
echo "  Generate:      python agent.py generate -s scripts/example.txt -c characters/portrait.jpg"
echo "  Web UI:        python agent.py serve"
echo "  Train:         python agent.py train -c characters/portrait.jpg"
echo ""
