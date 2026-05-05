"""Voice REPL — conversational, hands-free or push-to-talk interaction.

Two modes:

* ``push_to_talk``  — press ENTER, speak, ENTER to stop (or wait for VAD).
* ``hands_free``    — listens continuously, transcribes when it detects
                      voice activity, replies, then resumes listening.

Each utterance:
    mic ─► VAD-record ─► STT ─► cognitive graph ─► AIMessage ─► TTS ─► speaker

Optionally streams visible ASCII level-meter feedback so the user knows
the mic is live.

This module degrades gracefully — if neither STT nor TTS is available,
``run_voice_repl()`` prints a clear error and exits with code 2.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from isaac.core.graph import build_graph
from isaac.core.state import make_initial_state
from isaac.memory.context_manager import compress_messages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print(msg: str, *, color: str = "", end: str = "\n") -> None:
    if color and sys.stdout.isatty():
        sys.stdout.write(f"\033[{color}m{msg}\033[0m{end}")
    else:
        sys.stdout.write(msg + end)
    sys.stdout.flush()


def _level_meter(rms: float, width: int = 30) -> str:
    """Return an ASCII meter for an RMS amplitude (0..~0.3)."""
    norm = min(1.0, rms * 4)
    n = int(norm * width)
    return "[" + "#" * n + "-" * (width - n) + "]"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_voice_repl(hands_free: bool = False) -> int:
    """Launch the voice REPL.

    Parameters
    ----------
    hands_free:
        If True, don't wait for ENTER between utterances.  The mic stays
        open and VAD decides when to capture.

    Returns
    -------
    int
        Exit code (0 = normal, 2 = missing dependencies).
    """
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "apscheduler", "chromadb", "isaac"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # -- Capability check ---------------------------------------------------
    from isaac.multimodal.voice import is_stt_available, is_tts_available
    from isaac.multimodal.voice.audio_io import is_audio_available

    if not is_audio_available():
        _print(
            "Voice REPL: no audio device backend (install: pip install sounddevice soundfile).",
            color="31",
        )
        return 2
    if not is_stt_available():
        _print(
            "Voice REPL: no STT backend (install: pip install faster-whisper).",
            color="31",
        )
        return 2
    if not is_tts_available():
        _print(
            "Voice REPL: warning — no TTS backend; replies will be text-only.",
            color="33",
        )

    # -- Pre-warm --------------------------------------------------------
    from isaac.multimodal.voice.audio_io import (
        record_until_silence,
        save_wav,
    )
    from isaac.multimodal.voice.stt import get_stt
    from isaac.multimodal.voice.tts import get_tts

    _print("Loading STT model...", color="36")
    stt = get_stt()
    stt._load()  # eagerly load weights

    tts = get_tts() if is_tts_available() else None
    if tts is not None:
        _print("Loading TTS model...", color="36")
        try:
            tts._load()
        except Exception as exc:
            _print(f"TTS load failed: {exc} — falling back to text-only.", color="33")
            tts = None

    # -- Build graph + state ------------------------------------------------
    try:
        from isaac.tools import register_all_tools
        register_all_tools()
    except Exception:
        pass

    compiled = build_graph()
    state: dict[str, Any] = dict(make_initial_state())
    state["session_id"] = str(uuid.uuid4())

    _print("\n[I.S.A.A.C. — Voice REPL]", color="1;36")
    if hands_free:
        _print("Hands-free mode. Speak when ready.  Ctrl+C to exit.\n", color="36")
    else:
        _print("Push-to-talk mode. Press ENTER, speak, then stop talking.\n", color="36")

    # -- Loop --------------------------------------------------------------
    try:
        while True:
            if not hands_free:
                try:
                    input("[ENTER to record, /q to quit] ")
                except (EOFError, KeyboardInterrupt):
                    break

            _print("Listening... ", color="36", end="")
            sys.stdout.flush()

            def _meter(rms: float) -> None:
                sys.stdout.write("\r" + " " * 50 + "\rListening " + _level_meter(rms))
                sys.stdout.flush()

            try:
                audio = record_until_silence(on_chunk=_meter)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                _print(f"\nMicrophone error: {exc}", color="31")
                continue

            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()

            if len(audio) < 1000:  # < ~60 ms — assume nothing said
                _print("(nothing heard)", color="33")
                continue

            # STT
            t0 = time.monotonic()
            try:
                wav_path = save_wav(audio, f"/tmp/isaac_utt_{int(time.time())}.wav")
                user_text = stt.transcribe(wav_path)
            except Exception as exc:
                _print(f"STT error: {exc}", color="31")
                continue
            stt_ms = int((time.monotonic() - t0) * 1000)

            user_text = user_text.strip()
            if not user_text:
                _print("(no speech recognised)", color="33")
                continue

            _print(f">>> {user_text}  ({stt_ms} ms)", color="32")

            if user_text.strip().lower() in ("/q", "/quit", "/exit", "exit", "quit"):
                break

            # Sanitise
            try:
                from isaac.security.sanitizer import sanitize_input
                user_text = sanitize_input(user_text)
            except Exception:
                pass

            state["messages"] = [HumanMessage(content=user_text)]
            state["messages"] = compress_messages(state.get("messages", []))

            # Run the cognitive graph
            try:
                result: dict[str, Any] = {}
                for event in compiled.stream(dict(state)):
                    for _node, node_output in event.items():
                        if isinstance(node_output, dict):
                            result.update(node_output)
                state.update(result)

                # Extract the assistant's reply
                reply_text = ""
                for msg in result.get("messages", []):
                    if isinstance(msg, AIMessage):
                        c = msg.content
                        reply_text = c if isinstance(c, str) else str(c)
                if not reply_text:
                    reply_text = "I'm here, but I have nothing to say about that."

                _print(f"[I.S.A.A.C.] {reply_text}\n", color="1;36")

                # Speak
                if tts is not None:
                    try:
                        tts.speak(reply_text)
                    except Exception as exc:
                        _print(f"TTS playback failed: {exc}", color="33")

            except Exception as exc:
                logger.exception("Voice REPL: graph execution failed.")
                _print(f"[ERROR] {exc}\n", color="31")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            from isaac.nodes.computer_use import shutdown_ui_executor
            shutdown_ui_executor()
        except Exception:
            pass
        _print("\nGoodbye.", color="36")

    return 0
