#!/bin/sh
# entrypoint.sh — Executes /input/task.py and returns output.
#
# The task script is bind-mounted read-only by the host executor.
# stdout and stderr are captured by the Docker SDK.

set -e

if [ ! -f /input/task.py ]; then
    echo "ERROR: /input/task.py not found." >&2
    exit 1
fi

exec python /input/task.py
