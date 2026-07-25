"""HTTP client for a ComfyUI server.

Uses nothing but the standard library (``urllib``), so ``pip install
comfy-toolkit`` pulls in zero dependencies. WebSocket progress tracking is the
only optional extra.

The server address is resolved in this order:

1. the ``base_url`` argument,
2. the ``COMFY_URL`` environment variable,
3. ``http://127.0.0.1:8188``.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from . import results as _results
from .errors import (
    ComfyConnectionError,
    ComfyExecutionError,
    ComfySubmitError,
    ComfyTimeoutError,
)

__all__ = ["DEFAULT_BASE_URL", "ENV_VAR", "ComfyClient"]

#: Used when neither an explicit URL nor ``COMFY_URL`` is provided.
DEFAULT_BASE_URL = "http://127.0.0.1:8188"

#: Environment variable consulted for the server address.
ENV_VAR = "COMFY_URL"


def resolve_base_url(base_url: Optional[str] = None) -> str:
    """Resolve the server address from argument, environment, then default."""
    url = base_url or os.environ.get(ENV_VAR) or DEFAULT_BASE_URL
    return url.rstrip("/")


class ComfyClient:
    """A thin, synchronous client for ComfyUI's HTTP API.

    Example::

        client = ComfyClient()                       # or ComfyClient("http://gpu-box:8188")
        if not client.is_online():
            raise SystemExit("start ComfyUI first")
        prompt_id = client.submit(workflow)
        result = client.wait(prompt_id)

    The instance keeps a stable ``client_id``; ComfyUI uses it to route
    WebSocket events, so reusing one client across submissions is intentional.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        client_id: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Args:
            base_url: Server address. ``None`` falls back to ``COMFY_URL`` then
                ``http://127.0.0.1:8188``.
            client_id: Stable identifier for this client. Generated if omitted.
            timeout: Default socket timeout in seconds for HTTP calls.
        """
        self.base_url = resolve_base_url(base_url)
        self.client_id = client_id or str(uuid.uuid4())
        self.timeout = timeout

    def __repr__(self) -> str:
        return "ComfyClient(base_url={0!r})".format(self.base_url)

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _url(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def _request(self, path: str, *, data: Optional[bytes] = None,
                 method: str = "GET", timeout: Optional[float] = None,
                 params: Optional[Dict[str, Any]] = None) -> bytes:
        request = urllib.request.Request(self._url(path, params), data=data,
                                         method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(
                    request, timeout=timeout or self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # pragma: no cover - defensive
                pass
            raise ComfyConnectionError(
                "{0} {1} -> HTTP {2} {3}".format(method, path, exc.code, body)
            ) from exc
        except Exception as exc:
            raise ComfyConnectionError(
                "{0} {1} failed: {2}".format(method, path, exc)) from exc

    def get(self, path: str, *, params: Optional[Dict[str, Any]] = None,
            timeout: Optional[float] = None) -> Any:
        """GET *path* and decode the JSON response."""
        raw = self._request(path, method="GET", timeout=timeout, params=params)
        return json.loads(raw.decode("utf-8"))

    def post(self, path: str, payload: Dict, *,
             timeout: Optional[float] = None) -> Any:
        """POST a JSON *payload* to *path* and decode the JSON response."""
        raw = self._request(path, data=json.dumps(payload).encode("utf-8"),
                            method="POST", timeout=timeout)
        return json.loads(raw.decode("utf-8"))

    # ------------------------------------------------------------------
    # Health & status
    # ------------------------------------------------------------------

    def is_online(self, timeout: float = 3.0) -> bool:
        """Return ``True`` if the server answers ``/system_stats``."""
        try:
            self.get("/system_stats", timeout=timeout)
            return True
        except ComfyConnectionError:
            return False

    def system_stats(self, timeout: float = 5.0) -> Dict:
        """Return the raw ``/system_stats`` payload (Python, OS, devices)."""
        return self.get("/system_stats", timeout=timeout)

    def vram_stats(self, device_index: int = 0,
                   timeout: float = 5.0) -> Dict[str, Any]:
        """Return VRAM figures for one device, in bytes and GB.

        Returns:
            ``{"name", "vram_total", "vram_free", "vram_used",
            "vram_total_gb", "vram_free_gb", "vram_used_gb"}``. Values are
            ``0`` when the server reports no devices (CPU-only builds).
        """
        stats = self.system_stats(timeout=timeout)
        devices = stats.get("devices") or []
        device = devices[device_index] if device_index < len(devices) else {}
        total = int(device.get("vram_total") or 0)
        free = int(device.get("vram_free") or 0)
        gib = float(1024 ** 3)
        return {
            "name": device.get("name", "unknown"),
            "vram_total": total,
            "vram_free": free,
            "vram_used": max(total - free, 0),
            "vram_total_gb": round(total / gib, 2),
            "vram_free_gb": round(free / gib, 2),
            "vram_used_gb": round(max(total - free, 0) / gib, 2),
        }

    def queue_status(self, timeout: float = 3.0) -> Dict[str, int]:
        """Return ``{"running": N, "pending": N}``.

        Never raises: an unreachable server reads as an empty queue, which is
        what a status widget wants.
        """
        try:
            queue = self.get("/queue", timeout=timeout)
        except ComfyConnectionError:
            return {"running": 0, "pending": 0}
        return {
            "running": len(queue.get("queue_running") or []),
            "pending": len(queue.get("queue_pending") or []),
        }

    def interrupt(self, timeout: float = 5.0) -> None:
        """Ask the server to interrupt the job it is currently executing."""
        self._request("/interrupt", data=b"{}", method="POST", timeout=timeout)

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    @staticmethod
    def load_workflow(path: Union[str, Path]) -> Dict:
        """Load an API-format workflow JSON file from disk.

        Export it from ComfyUI with *Save (API Format)* — the canvas format is
        a different, incompatible schema.
        """
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def submit(self, workflow: Dict, *, client_id: Optional[str] = None,
               timeout: float = 30.0) -> str:
        """Queue a workflow and return its ``prompt_id``.

        Raises:
            ComfySubmitError: The server rejected the graph or returned no id.
            ComfyConnectionError: The server was unreachable.
        """
        payload = {"prompt": workflow, "client_id": client_id or self.client_id}
        try:
            response = self.post("/prompt", payload, timeout=timeout)
        except ComfyConnectionError as exc:
            raise ComfySubmitError(
                "ComfyUI rejected the workflow: {0}".format(exc)) from exc

        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise ComfySubmitError(
                "ComfyUI returned no prompt_id: {0!r}".format(response))
        return prompt_id

    # ------------------------------------------------------------------
    # History & results
    # ------------------------------------------------------------------

    def history(self, prompt_id: str, timeout: float = 5.0) -> Optional[Dict]:
        """Return the raw history entry for *prompt_id*, or ``None``.

        ``None`` means "not there yet" — either still queued, or the entry has
        not been flushed. See :func:`comfy_toolkit.submit_and_track` for why
        that distinction matters.
        """
        try:
            history = self.get(
                "/history/{0}".format(urllib.parse.quote(str(prompt_id))),
                timeout=timeout)
        except ComfyConnectionError:
            return None
        if not history or prompt_id not in history:
            return None
        return history[prompt_id]

    def result(self, prompt_id: str, timeout: float = 5.0) -> Optional[Dict]:
        """Return a normalized result dict for *prompt_id*, or ``None``.

        The dict has ``status`` (``completed`` / ``running`` / ``error``),
        ``images``, ``videos`` and ``filenames``. Video nodes are read from the
        ``gifs`` / ``videos`` output keys, not ``images``.
        """
        entry = self.history(prompt_id, timeout=timeout)
        if entry is None:
            return None
        return _results.parse_history_entry(entry)

    def wait(
        self,
        prompt_id: str,
        *,
        expect: str = "any",
        max_wait: float = 600.0,
        poll_interval: float = 2.0,
        empty_grace: float = 5.0,
        on_poll: Optional[Callable[[float, Optional[Dict]], None]] = None,
    ) -> Dict:
        """Block until *prompt_id* produces output.

        Args:
            prompt_id: Id returned by :meth:`submit`.
            expect: ``"images"``, ``"videos"`` or ``"any"``. Completion is only
                accepted once the requested kind of output exists, which avoids
                returning an empty result the instant the status flag flips.
            max_wait: Seconds before giving up.
            poll_interval: Seconds between ``/history`` polls.
            empty_grace: With ``expect="any"``, how long to keep polling after
                the server says "completed" but the outputs list is still
                empty. Covers the flush lag; after it elapses, the empty result
                is returned as-is (some workflows really do save nothing).
            on_poll: Called as ``on_poll(elapsed, result_or_none)`` each poll.

        Returns:
            The normalized result dict.

        Raises:
            ComfyExecutionError: The server reported a failure.
            ComfyTimeoutError: *max_wait* elapsed first.
        """
        deadline = time.time() + max_wait
        empty_since: Optional[float] = None
        while True:
            result = self.result(prompt_id)
            if on_poll is not None:
                on_poll(max_wait - max(deadline - time.time(), 0.0), result)
            if result is not None:
                if result["status"] == "error":
                    raise ComfyExecutionError(
                        "generation failed for {0}".format(prompt_id),
                        prompt_id=prompt_id, detail=result.get("error", ""))
                if result["status"] == "completed":
                    if _has_expected(result, expect):
                        return result
                    if expect == "any":
                        now = time.time()
                        if empty_since is None:
                            empty_since = now
                        elif now - empty_since >= empty_grace:
                            return result
            if time.time() >= deadline:
                raise ComfyTimeoutError(
                    "timed out after {0:.0f}s waiting for {1}".format(
                        max_wait, prompt_id),
                    prompt_id=prompt_id)
            time.sleep(poll_interval)

    def wait_for_images(self, prompt_id: str, **kwargs) -> List[Dict]:
        """Block until images exist and return their descriptors."""
        kwargs.setdefault("expect", "images")
        return self.wait(prompt_id, **kwargs)["images"]

    def wait_for_videos(self, prompt_id: str, **kwargs) -> List[Dict]:
        """Block until video output exists and return its descriptors.

        Defaults to a longer wait than :meth:`wait_for_images`, since video
        jobs routinely run for many minutes.
        """
        kwargs.setdefault("expect", "videos")
        kwargs.setdefault("max_wait", 1800.0)
        kwargs.setdefault("poll_interval", 3.0)
        return self.wait(prompt_id, **kwargs)["videos"]

    # ------------------------------------------------------------------
    # Fetching generated files
    # ------------------------------------------------------------------

    def download(self, descriptor: Dict, timeout: float = 60.0) -> bytes:
        """Fetch one generated file's bytes through ``/view``.

        Works with a remote server, unlike copying from ComfyUI's output
        folder. Pass a descriptor from ``result["images"]`` or
        ``result["videos"]``.
        """
        params = _results.output_url_params(descriptor)
        return self._request("/view", method="GET", timeout=timeout,
                             params=params)

    def save(self, descriptor: Dict, dest: Union[str, Path],
             timeout: float = 60.0) -> Path:
        """Download one generated file and write it to *dest*.

        If *dest* is an existing directory, the server-side filename is used
        inside it.
        """
        target = Path(dest)
        if target.is_dir():
            target = target / Path(descriptor.get("filename", "output")).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.download(descriptor, timeout=timeout))
        return target

    # ------------------------------------------------------------------
    # Progress tracking (optional extra)
    # ------------------------------------------------------------------

    def ws_url(self, client_id: Optional[str] = None) -> str:
        """Return the WebSocket URL for this server and client id."""
        base = self.base_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        return "{0}/ws?clientId={1}".format(
            base, urllib.parse.quote(client_id or self.client_id))

    def submit_and_track(self, workflow: Dict, **kwargs) -> Dict:
        """Submit *workflow* and follow it over WebSocket to completion.

        See :func:`comfy_toolkit.tracking.submit_and_track` for the full
        signature. Requires the ``ws`` extra.
        """
        from .tracking import submit_and_track
        return submit_and_track(self, workflow, **kwargs)


def _has_expected(result: Dict, expect: str) -> bool:
    """Return whether *result* carries the kind of output the caller wants."""
    if expect == "images":
        return bool(result.get("images"))
    if expect == "videos":
        return bool(result.get("videos"))
    return bool(result.get("images") or result.get("videos"))
