#!/usr/bin/env bash
# I.S.A.A.C. Installer — curl | bash
# Auto-provisions Docker, Ollama, and the default model, then installs isaac-agent.

set -euo pipefail

# --- Constants ---
readonly ISAAC_VERSION="1.6.2"
readonly DEFAULT_MODEL="qwen3.6"
readonly REPO_OWNER="Matheussoranco"
readonly REPO_NAME="I.S.A.A.C"

# --- Helpers ---
info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; }
die()   { error "$*"; exit 1; }

# --- OS/Arch detection ---
detect_os() {
    case "$(uname -s)" in
        Linux)  OS="linux" ;;
        Darwin) OS="darwin" ;;
        CYGWIN*|MINGW*|MSYS*) OS="windows" ;;
        *) die "Unsupported OS: $(uname -s)" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) ARCH="x86_64" ;;
        arm64|aarch64) ARCH="arm64" ;;
        *) die "Unsupported architecture: $(uname -m)" ;;
    esac
}

# --- Docker ---
ensure_docker() {
    if command -v docker >/dev/null 2>&1; then
        info "Docker already installed: $(docker --version)"
        return 0
    fi

    info "Installing Docker..."
    case "$OS" in
        linux)
            # Try apt (Debian/Ubuntu), then dnf (Fedora), then pacman (Arch)
            if command -v apt >/dev/null 2>&1; then
                sudo apt update && sudo apt install -y docker.io
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y docker
            elif command -v pacman >/dev/null 2>&1; then
                sudo pacman -S --noconfirm docker
            else
                die "No supported package manager found. Install Docker manually."
            fi
            sudo systemctl enable --now docker
            sudo usermod -aG docker "$USER"
            warn "You may need to log out and back in for docker group changes to take effect."
            ;;
        darwin)
            if command -v brew >/dev/null 2>&1; then
                brew install --cask docker
            else
                die "Homebrew not found. Install Docker Desktop manually from https://docker.com"
            fi
            ;;
        windows)
            die "On Windows, please install Docker Desktop manually from https://docker.com"
            ;;
    esac
}

# --- Ollama ---
ensure_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        info "Ollama already installed: $(ollama --version)"
        return 0
    fi

    info "Installing Ollama..."
    case "$OS" in
        linux)
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        darwin)
            if command -v brew >/dev/null 2>&1; then
                brew install ollama
            else
                curl -fsSL https://ollama.com/install.sh | sh
            fi
            ;;
        windows)
            die "On Windows, please install Ollama manually from https://ollama.com"
            ;;
    esac
}

start_ollama() {
    info "Starting Ollama service..."
    case "$OS" in
        linux)
            sudo systemctl enable --now ollama 2>/dev/null || ollama serve &
            ;;
        darwin)
            brew services start ollama 2>/dev/null || ollama serve &
            ;;
        *)
            ollama serve &
            ;;
    esac

    # Wait for Ollama to be ready
    local retries=30
    while ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
        ((retries--)) || die "Ollama failed to start within 30 seconds"
        sleep 1
    done
    info "Ollama is ready."
}

pull_model() {
    local model="${1:-$DEFAULT_MODEL}"
    info "Pulling model: $model"
    ollama pull "$model" || die "Failed to pull model $model"
}

# --- Python / pip ---
ensure_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        die "Python 3.12+ not found. Install Python first."
    fi
    local version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "${version%.*}" -lt 3 ]] || [[ "${version#*.}" -lt 12 ]]; then
        die "Python 3.12+ required, found $version"
    fi
    info "Python $version found."
}

ensure_pip() {
    python3 -m pip --version >/dev/null 2>&1 || die "pip not found"
}

install_isaac() {
    info "Installing isaac-agent v$ISAAC_VERSION..."
    python3 -m pip install --upgrade pip
    python3 -m pip install "isaac-agent==$ISAAC_VERSION"
    info "isaac-agent installed successfully."
}

# --- Post-install ---
run_doctor() {
    info "Running isaac doctor --auto-fix..."
    isaac doctor --auto-fix || warn "isaac doctor reported issues (see output above)."
    info "Installation complete! Run 'isaac --help' to get started."
}

# --- Main ---
main() {
    info "I.S.A.A.C. Installer v$ISAAC_VERSION"
    info "Detecting platform..."
    detect_os
    detect_arch
    info "OS: $OS, Arch: $ARCH"

    ensure_python
    ensure_pip
    ensure_docker
    ensure_ollama
    start_ollama
    pull_model
    install_isaac
    run_doctor
}

main "$@"