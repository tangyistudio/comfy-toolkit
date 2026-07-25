"""Exception hierarchy for comfy-toolkit.

Everything raised by this package derives from :class:`ComfyToolkitError`, so a
single ``except ComfyToolkitError`` is enough to contain a generation pipeline.
"""

__all__ = [
    "ComfyToolkitError",
    "ComfyConnectionError",
    "ComfySubmitError",
    "ComfyExecutionError",
    "ComfyTimeoutError",
    "HistoryNotReadyError",
    "NodeNotFoundError",
    "WebSocketUnavailableError",
]


class ComfyToolkitError(Exception):
    """Base class for every error raised by comfy-toolkit."""


class ComfyConnectionError(ComfyToolkitError):
    """The ComfyUI server could not be reached."""


class ComfySubmitError(ComfyToolkitError):
    """The server rejected a workflow, or returned no ``prompt_id``."""


class ComfyExecutionError(ComfyToolkitError):
    """ComfyUI reported ``execution_error`` / ``execution_interrupted``.

    Attributes:
        prompt_id: The prompt that failed, when known.
        detail: Raw payload from the server, truncated for readability.
    """

    def __init__(self, message: str, prompt_id: str = "", detail: str = ""):
        super().__init__(message)
        self.prompt_id = prompt_id
        self.detail = detail


class ComfyTimeoutError(ComfyToolkitError, TimeoutError):
    """A wait loop gave up before the job finished."""

    def __init__(self, message: str, prompt_id: str = ""):
        super().__init__(message)
        self.prompt_id = prompt_id


class HistoryNotReadyError(ComfyTimeoutError):
    """Execution finished but ``/history/{prompt_id}`` never produced outputs.

    This is the race condition described in the README: the WebSocket channel
    announces completion before the history entry has been flushed. It is
    raised only after the configured number of retries has been exhausted.
    """


class NodeNotFoundError(ComfyToolkitError):
    """A workflow did not contain a node the caller needs."""


class WebSocketUnavailableError(ComfyToolkitError):
    """Progress tracking was requested but ``websocket-client`` is missing.

    Install the optional extra::

        pip install "comfy-toolkit[ws]"
    """
