#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# I.S.A.A.C. UI Sandbox Entrypoint
# Boots virtual display → window manager → optional VNC → runs script or idle
# ---------------------------------------------------------------------------
set -e

export DISPLAY="${DISPLAY:-:99}"

# 1. Start virtual framebuffer display
Xvfb "${DISPLAY}" \
    -screen 0 "${SCREEN_WIDTH:-1280}x${SCREEN_HEIGHT:-720}x${SCREEN_DEPTH:-24}" \
    -ac \
    +extension GLX \
    +render \
    -noreset \
    &
XVFB_PID=$!

# Wait for display to be ready
sleep 0.8
for i in $(seq 1 10); do
    if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
        break
    fi
    sleep 0.3
done

# 2. Start lightweight window manager (needed for window focus events)
openbox --sm-disable &

# 3. Start VNC server if ISAAC_VNC_ENABLED is set (for debugging only)
if [ "${ISAAC_VNC_ENABLED:-0}" = "1" ]; then
    x11vnc \
        -display "${DISPLAY}" \
        -forever \
        -nopw \
        -rfbport "${VNC_PORT:-5900}" \
        -bg \
        -quiet \
        -noxdamage
fi

# 4. If a script argument is provided, run it and exit
if [ -n "$1" ]; then
    exec python3 "$@"
fi

# 5. Keep the container alive for interactive computer-use sessions
# (I.S.A.A.C. will exec commands via docker exec)
wait ${XVFB_PID}
