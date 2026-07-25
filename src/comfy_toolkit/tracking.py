"""WebSocket progress tracking, and the ``/history`` race condition fix.

The race
--------
ComfyUI announces completion on its WebSocket channel *before* the matching
``/history/{prompt_id}`` entry is guaranteed to be readable. The obvious code::

    ws.recv()                      # "execution_success"
    result = client.result(pid)    # None, or completed-with-zero-images

fails intermittently — more often on big images, video encodes, slow disks, and
network shares, because those widen the window between "the executor is done"
and "the history entry has been written". It is a classic
time-of-check/time-of-use race, and it is why so many ComfyUI integrations
carry a mysterious ``sleep(2)``.

The fix
-------
Treat the WebSocket message as a *hint*, never as the result. After the socket
reports completion, poll ``/history`` up to :data:`HISTORY_RETRY_ATTEMPTS`
times with a short delay, and only accept an entry that actually contains
output files. A fixed sleep cannot be tuned correctly: too short and it still
races, too long and every generation pays for the worst case. Retrying until
outputs appear costs nothing on the happy path and is correct on the slow one.

Only if every retry comes back empty is :class:`~comfy_toolkit.errors.
HistoryNotReadyError` raised — a real failure, not a guess.
"""

import json
import time
from typing import Callable, Dict, Optional

from .errors import (
    ComfyExecutionError,
    ComfyTimeoutError,
    HistoryNotReadyError,
    WebSocketUnavailableError,
)

__all__ = [
    "HISTORY_RETRY_ATTEMPTS",
    "HISTORY_RETRY_DELAY",
    "submit_and_track",
]

#: How many times to re-read ``/history`` after the socket says "done".
HISTORY_RETRY_ATTEMPTS = 20

#: Seconds between those retries. 20 x 1.0s gives a ~20 second budget, which
#: comfortably covers video muxing on a spinning disk.
HISTORY_RETRY_DELAY = 1.0


def _require_websocket():
    try:
        import websocket  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise WebSocketUnavailableError(
            "progress tracking needs the 'websocket-client' package; "
            "install it with: pip install \"comfy-toolkit[ws]\""
        ) from exc
    return websocket


def _fetch_result(client, prompt_id: str, expect: str) -> Optional[Dict]:
    """Return a usable result dict, or ``None`` if it is not ready yet."""
    result = client.result(prompt_id)
    if result is None:
        return None
    if result["status"] == "error":
        raise ComfyExecutionError(
            "generation failed for {0}".format(prompt_id),
            prompt_id=prompt_id, detail=result.get("error", ""))
    if result["status"] != "completed":
        return None

    if expect == "images":
        return result if result["images"] else None
    if expect == "videos":
        return result if result["videos"] else None
    return result if (result["images"] or result["videos"]) else None


def submit_and_track(
    client,
    workflow: Dict,
    *,
    on_progress: Optional[Callable[[int, int], None]] = None,
    on_node: Optional[Callable[[Optional[str]], None]] = None,
    client_id: Optional[str] = None,
    expect: str = "images",
    overall_timeout: float = 600.0,
    recv_timeout: float = 5.0,
    history_retries: int = HISTORY_RETRY_ATTEMPTS,
    history_retry_delay: float = HISTORY_RETRY_DELAY,
) -> Dict:
    """Submit a workflow, stream step progress, and return the finished result.

    Requires the optional ``ws`` extra (``pip install "comfy-toolkit[ws]"``).

    The socket is opened *before* the workflow is submitted. That ordering
    matters: submit first and a fast job can finish before the socket is
    listening, so no progress events are ever seen.

    Args:
        client: A :class:`~comfy_toolkit.client.ComfyClient`.
        workflow: API-format workflow dict, ready to run.
        on_progress: Called as ``on_progress(step, total)`` on every sampler
            step. Exceptions raised inside the callback are swallowed so a
            broken progress bar cannot kill a generation.
        on_node: Called as ``on_node(node_id)`` when execution moves to a new
            node, and with ``None`` when the graph finishes.
        client_id: Override the client's identifier for this run.
        expect: ``"images"``, ``"videos"`` or ``"any"`` — which outputs must be
            present before the result counts as ready.
        overall_timeout: Seconds to spend on the socket phase.
        recv_timeout: Socket read timeout. On each read timeout the history
            endpoint is checked as a backstop, so a dropped socket degrades to
            polling instead of hanging.
        history_retries: How many times to re-read ``/history`` after the
            socket reports completion. See the module docstring.
        history_retry_delay: Seconds between those retries.

    Returns:
        The normalized result dict, with a ``prompt_id`` key added.

    Raises:
        WebSocketUnavailableError: ``websocket-client`` is not installed.
        ComfyExecutionError: ComfyUI reported an execution error.
        ComfyTimeoutError: *overall_timeout* elapsed during execution.
        HistoryNotReadyError: Execution finished but no outputs ever appeared.
    """
    websocket = _require_websocket()
    resolved_client_id = client_id or client.client_id

    socket = websocket.WebSocket()
    try:
        socket.connect(client.ws_url(resolved_client_id), timeout=10)
    except Exception as exc:
        raise ComfyTimeoutError(
            "could not open WebSocket to {0}: {1}".format(client.base_url, exc)
        ) from exc

    try:
        prompt_id = client.submit(workflow, client_id=resolved_client_id)
    except Exception:
        _close(socket)
        raise

    socket.settimeout(recv_timeout)
    deadline = time.time() + overall_timeout
    finished = False
    failure: Optional[str] = None

    try:
        while time.time() < deadline:
            try:
                raw = socket.recv()
            except Exception:
                # Read timed out or the socket dropped. Fall back to polling
                # so a flaky connection cannot strand the caller.
                early = _fetch_result(client, prompt_id, expect)
                if early is not None:
                    early["prompt_id"] = prompt_id
                    return early
                continue

            if isinstance(raw, (bytes, bytearray)):
                # Binary frames carry preview images; not our concern here.
                continue
            try:
                message = json.loads(raw)
            except Exception:
                continue

            message_type = message.get("type")
            data = message.get("data") or {}
            message_prompt_id = data.get("prompt_id")

            if message_type == "progress":
                value, maximum = data.get("value"), data.get("max")
                if (on_progress is not None and isinstance(value, int)
                        and isinstance(maximum, int) and maximum > 0):
                    try:
                        on_progress(value, maximum)
                    except Exception:
                        pass

            elif message_type == "executing":
                node = data.get("node")
                if on_node is not None:
                    try:
                        on_node(node)
                    except Exception:
                        pass
                # node=None means "the graph is done" for this prompt.
                if node is None and message_prompt_id == prompt_id:
                    finished = True
                    break

            elif message_type == "execution_success":
                if message_prompt_id == prompt_id:
                    finished = True
                    break

            elif message_type in ("execution_error", "execution_interrupted"):
                if message_prompt_id == prompt_id:
                    failure = str(data)[:500]
                    break
    finally:
        _close(socket)

    if failure is not None:
        raise ComfyExecutionError(
            "ComfyUI reported an execution error",
            prompt_id=prompt_id, detail=failure)

    if not finished:
        raise ComfyTimeoutError(
            "timed out after {0:.0f}s tracking {1}".format(
                overall_timeout, prompt_id),
            prompt_id=prompt_id)

    # --- The race fix ------------------------------------------------------
    # The socket said "done". The history entry may not be readable yet, so
    # retry until it carries real outputs instead of trusting the first read.
    for attempt in range(max(history_retries, 1)):
        result = _fetch_result(client, prompt_id, expect)
        if result is not None:
            result["prompt_id"] = prompt_id
            return result
        if attempt < history_retries - 1:
            time.sleep(history_retry_delay)

    raise HistoryNotReadyError(
        "execution finished but /history never returned {0} for {1} "
        "(waited {2:.0f}s)".format(
            expect, prompt_id, history_retries * history_retry_delay),
        prompt_id=prompt_id,
    )


def _close(socket) -> None:
    try:
        socket.close()
    except Exception:
        pass
