"""Live step progress over WebSocket, with the /history race handled.

Requires the optional extra:
    pip install "comfy-toolkit[ws]"

Usage:
    python examples/02_progress_tracking.py path/to/workflow_api.json

submit_and_track() does not trust the WebSocket "done" message as a result.
After the socket reports completion it re-reads /history until real output
files appear, because ComfyUI announces completion before that entry is
guaranteed to be readable.
"""

import sys

from comfy_toolkit import ComfyClient, customize
from comfy_toolkit.errors import (
    ComfyExecutionError,
    HistoryNotReadyError,
    WebSocketUnavailableError,
)


def render_bar(step: int, total: int, width: int = 30) -> str:
    filled = int(width * step / total)
    return "[{0}{1}] {2}/{3}".format("#" * filled, "." * (width - filled),
                                     step, total)


def main(workflow_path: str) -> int:
    client = ComfyClient()
    if not client.is_online():
        print("ComfyUI is not reachable. Start it, or set COMFY_URL.")
        return 1

    workflow = customize(
        client.load_workflow(workflow_path),
        positive="a snowy pine forest at blue hour",
        negative="blurry, low quality",
        filename_prefix="example_progress",
    )

    def on_progress(step, total):
        print("\r" + render_bar(step, total), end="", flush=True)

    def on_node(node_id):
        if node_id is None:
            print("\ngraph finished, waiting for outputs to be written...")

    try:
        result = client.submit_and_track(
            workflow,
            on_progress=on_progress,
            on_node=on_node,
            expect="images",
            overall_timeout=600,
        )
    except WebSocketUnavailableError as exc:
        print(exc)
        return 1
    except HistoryNotReadyError as exc:
        # Every retry came back empty: a genuine failure, not the race.
        print("\n", exc)
        return 1
    except ComfyExecutionError as exc:
        print("\nexecution failed:", exc.detail)
        return 1

    print("done:", result["filenames"])
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
