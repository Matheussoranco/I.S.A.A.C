# I.S.A.A.C. Agent — production container (multi-arch)
#
# Build:  docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/matheussoranco/isaac-agent:1.6.2 --push .
# Run:    docker run --rm -it ghcr.io/matheussoranco/isaac-agent:1.6.2

# Syntax: https://docs.docker.com/engine/reference/builder/
# Use buildx for multi-arch: docker buildx build --platform linux/amd64,linux/arm64 ...

# ---- Base stage ----
FROM python:3.12-slim AS base

# System deps + Docker CLI (needed for Docker SDK to spawn sandboxes)
RUN apt-get update     && apt-get install -y --no-install-recommends         curl         ca-certificates         gnupg     && install -m 0755 -d /etc/apt/keyrings     && curl -fsSL https://download.docker.com/linux/debian/gpg        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg]        https://download.docker.com/linux/debian bookworm stable"        > /etc/apt/sources.list.d/docker.list     && apt-get update     && apt-get install -y --no-install-recommends docker-ce-cli     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layer cached)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY pyproject.toml README.md LICENSE NOTICE /app/
COPY src/ /app/src/
COPY skills/ /app/skills/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create Isaac home directory
RUN mkdir -p /root/.isaac/workspace

# Expose nothing by default — all comms go through Telegram or CLI
ENTRYPOINT ["python", "-m", "isaac"]
CMD ["run"]
