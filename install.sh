#!/usr/bin/env bash
# =============================================================
#  Hidra-TTS — Script de Instalación (sin Docker)
#  Requisitos: Ubuntu 22.04+, GPU NVIDIA con CUDA 12+
# =============================================================
set -euo pipefail

# ── Colores ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Directorio raíz del proyecto ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_VERSION="3.11"

echo -e "\n${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║        Hidra-TTS  ·  Instalador          ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}\n"

cd "$SCRIPT_DIR"

# ── Detective de Sudo ─────────────────────────────────────────
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        warn "No eres root y 'sudo' no está instalado. El script podría fallar en pasos del sistema."
    fi
fi

# ── 1. Dependencias del sistema ───────────────────────────────
info "Instalando dependencias del sistema..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    software-properties-common curl git ffmpeg libsndfile1 sox

# Python 3.11 via deadsnakes
if ! python3.11 --version &>/dev/null 2>&1; then
    info "Agregando PPA deadsnakes e instalando Python ${PYTHON_VERSION}..."
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq \
        python3.11 python3.11-venv python3.11-dev python3.11-distutils
else
    success "Python ${PYTHON_VERSION} ya instalado: $(python3.11 --version)"
fi

# ── 2. Entorno virtual ────────────────────────────────────────
info "Creando entorno virtual en ${VENV_DIR}..."
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
success "Entorno virtual activado."

# ── 3. Actualizar pip ─────────────────────────────────────────
info "Actualizando pip..."
pip install --quiet --upgrade pip setuptools wheel hf_transfer

# ── 4. PyTorch con soporte CUDA ───────────────────────────────
# Detección de Blackwell (sm_120 -> RTX 5090)
TORCH_URL="https://download.pytorch.org/whl/cu124"
TORCH_VERSION="torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"

if command -v nvidia-smi >/dev/null 2>&1; then
    # Intenta obtener el major compute capability (p.ej. 12 para sm_120)
    # nvmlDeviceGetMaxPcieLinkGeneration no es confiable, mejor nvidia-smi controlando arch
    # Si detectamos una arquitectura Blackwell (120 o superior), usamos Nightly
    # (En RunPod nvidia-smi suele estar disponible)
    if python3 -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null | grep -q "^1[0-9]"; then
        info "Detectada GPU Blackwell (sm_120+). Instalando PyTorch 2.7+ Nightly (cu128) para compatibilidad..."
        TORCH_URL="https://download.pytorch.org/whl/nightly/cu128"
        TORCH_VERSION="--pre torch torchvision torchaudio"
    fi
fi

info "Instalando PyTorch (${TORCH_VERSION}) desde ${TORCH_URL}..."
pip install --quiet ${TORCH_VERSION} --index-url ${TORCH_URL}
success "PyTorch instalado."

# ── 5. Instalar el proyecto y sus dependencias ────────────────
info "Instalando hidra-tts y sus dependencias desde pyproject.toml..."
pip install --quiet -e .
success "hidra-tts instalado."

# ── 6. Script de arranque del servidor ───────────────────────
info "Generando script de arranque: run_server.sh..."
cat > "$SCRIPT_DIR/run_server.sh" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
cd "$SCRIPT_DIR"

# ── Blackwell (sm_120) compatibility ──────────────────────────────────────────
# Disable HuggingFace Hub pre-compiled CUDA kernels (e.g. RMSNorm) which have
# no kernel image for sm_120 (RTX 5080/5090). Falls back to native PyTorch.
export TRANSFORMERS_USE_KERNELS=0

echo "🚀 Iniciando Hidra-TTS server..."
python main.py
EOF
chmod +x "$SCRIPT_DIR/run_server.sh"
success "run_server.sh creado."

# ── Resumen ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ✅  Instalación completada con éxito    ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  Para iniciar el servidor:               ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    ${GREEN}./run_server.sh${NC}                        ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}                                          ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  O manualmente:                          ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    ${GREEN}source .venv/bin/activate${NC}             ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    ${GREEN}python main.py${NC}                         ${BOLD}║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
