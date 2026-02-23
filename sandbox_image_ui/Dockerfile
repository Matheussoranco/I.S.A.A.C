FROM ubuntu:22.04

LABEL maintainer="I.S.A.A.C. Project"
LABEL description="Virtual desktop sandbox with Xvfb + VNC + xdotool for Computer-Use tasks"

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    SCREEN_WIDTH=1280 \
    SCREEN_HEIGHT=720 \
    SCREEN_DEPTH=24 \
    VNC_PORT=5900 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    # Virtual display & window manager
    xvfb \
    openbox \
    x11-utils \
    x11-xserver-utils \
    # UI automation tools
    xdotool \
    scrot \
    imagemagick \
    # VNC server (for optional remote viewing / debugging)
    x11vnc \
    # Accessibility
    at-spi2-core \
    # Browser
    chromium-browser \
    # Python runtime
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    libffi-dev \
    libgomp1 \
    # Locale
    locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python packages for UI automation inside the sandbox
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/sandbox_requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/sandbox_requirements.txt

# Install Playwright browsers (Chromium only)
RUN playwright install chromium --with-deps 2>/dev/null || true

# ---------------------------------------------------------------------------
# Non-root user (UID 65534 = nobody — same as code sandbox for consistency)
# ---------------------------------------------------------------------------
RUN useradd --uid 1000 --create-home --shell /bin/bash sandboxuser
USER 1000
WORKDIR /workspace

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
COPY entrypoint_ui.sh /entrypoint_ui.sh
ENTRYPOINT ["/entrypoint_ui.sh"]
