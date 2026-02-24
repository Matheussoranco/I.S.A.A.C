# I.S.A.A.C. Agent — production container
#
# Build:  docker compose build isaac
# Run:    docker compose up isaac

FROM python:3.12-slim AS base

# System deps + Docker CLI (needed for Docker SDK to spawn sandboxes)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian bookworm stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layer cached)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY skills/ /app/skills/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create Isaac home directory
RUN mkdir -p /root/.isaac/workspace

# Expose nothing by default — all comms go through Telegram or CLI
ENTRYPOINT ["python", "-m", "isaac"]
CMD ["run"]
