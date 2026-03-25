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

# ── 1. Dependencias del sistema ───────────────────────────────
info "Instalando dependencias del sistema (requiere sudo)..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    software-properties-common curl git ffmpeg libsndfile1 sox

# Python 3.11 via deadsnakes
if ! python3.11 --version &>/dev/null 2>&1; then
    info "Agregando PPA deadsnakes e instalando Python ${PYTHON_VERSION}..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
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

# ── 4. PyTorch con soporte CUDA 12.4 ─────────────────────────
info "Instalando PyTorch 2.6.0 + torchaudio + torchvision (CUDA 12.4)..."
pip install --quiet \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
success "PyTorch instalado."

# ── 5. Flash-Attention (rueda precompilada) ───────────────────
info "Instalando flash-attention (rueda precompilada para PyTorch 2.6 + CUDA 12 + Python 3.11)..."
pip install --quiet \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
success "flash-attention instalado."

# ── 6. Instalar el proyecto y sus dependencias ────────────────
info "Instalando hidra-tts y sus dependencias desde pyproject.toml..."
pip install --quiet -e .
success "hidra-tts instalado."

# ── 7. Script de arranque del servidor ───────────────────────
info "Generando script de arranque: run_server.sh..."
cat > "$SCRIPT_DIR/run_server.sh" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
cd "$SCRIPT_DIR"
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
