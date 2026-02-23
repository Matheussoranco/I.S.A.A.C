# I.S.A.A.C. Sandbox — minimal, unprivileged Python execution environment.
#
# Build:  docker build -t isaac-sandbox:latest sandbox_image/
# Usage:  Managed by isaac.sandbox.manager — never run manually.

FROM python:3.12-slim AS base

# Harden: remove apt cache + unnecessary packages
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install whitelisted Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Create non-root user (UID 65534 = nobody)
RUN useradd --uid 65534 --no-create-home --shell /usr/sbin/nologin sandbox \
    || true

# Working directory for task execution
WORKDIR /workspace

# Default entrypoint: execute the task script injected at /input/task.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER 65534:65534

ENTRYPOINT ["/entrypoint.sh"]
