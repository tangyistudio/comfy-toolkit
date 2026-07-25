"""comfy-toolkit — a small, dependency-free client for ComfyUI's HTTP API.

Three problems it exists to solve:

1. **The ``/history`` race.** ComfyUI's WebSocket says "done" before the
   history entry is readable. :func:`submit_and_track` retries until real
   outputs appear instead of sleeping and hoping.
2. **LoRA stacking.** :func:`inject_loras` appends only the LoRA loaders a
   request actually needs and rewires the graph, instead of pre-wiring
   everything at strength 0 and degrading quality.
3. **Video outputs.** ``VHS_VideoCombine`` writes to ``gifs`` / ``videos``, not
   ``images``. :func:`extract_videos` reads both.

Quickstart::

    from comfy_toolkit import ComfyClient, customize

    client = ComfyClient()                       # or COMFY_URL env var
    workflow = client.load_workflow("my_workflow_api.json")
    workflow = customize(workflow, positive="a lighthouse at dawn", seed=42)
    result = client.submit_and_track(workflow, on_progress=print)
    print(result["filenames"])
"""

from .client import DEFAULT_BASE_URL, ENV_VAR, ComfyClient, resolve_base_url
from .errors import (
    ComfyConnectionError,
    ComfyExecutionError,
    ComfySubmitError,
    ComfyTimeoutError,
    ComfyToolkitError,
    HistoryNotReadyError,
    NodeNotFoundError,
    WebSocketUnavailableError,
)
from .inject import (
    customize,
    find_model_chain_tail,
    inject_loras,
    random_seed,
    set_filename_prefix,
    set_input_image,
    set_lora_strengths_by_title,
    set_prompts,
    set_seed,
)
from .nodes import (
    classify_text_encode_nodes,
    find_load_image_nodes,
    find_lora_nodes,
    find_node_by_class,
    find_node_by_title,
    find_nodes_by_class,
    find_nodes_by_title,
    find_prompt_nodes,
    find_sampler_nodes,
    find_text_encode_nodes,
    next_node_id,
)
from .results import (
    collect_outputs,
    extract_images,
    extract_videos,
    output_url_params,
    parse_history_entry,
    write_sidecar,
)
from .tracking import (
    HISTORY_RETRY_ATTEMPTS,
    HISTORY_RETRY_DELAY,
    submit_and_track,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # client
    "ComfyClient",
    "resolve_base_url",
    "DEFAULT_BASE_URL",
    "ENV_VAR",
    # tracking
    "submit_and_track",
    "HISTORY_RETRY_ATTEMPTS",
    "HISTORY_RETRY_DELAY",
    # inject
    "customize",
    "inject_loras",
    "set_lora_strengths_by_title",
    "set_prompts",
    "set_seed",
    "set_filename_prefix",
    "set_input_image",
    "find_model_chain_tail",
    "random_seed",
    # nodes
    "find_nodes_by_class",
    "find_node_by_class",
    "find_nodes_by_title",
    "find_node_by_title",
    "find_sampler_nodes",
    "find_text_encode_nodes",
    "classify_text_encode_nodes",
    "find_prompt_nodes",
    "find_load_image_nodes",
    "find_lora_nodes",
    "next_node_id",
    # results
    "extract_images",
    "extract_videos",
    "parse_history_entry",
    "output_url_params",
    "write_sidecar",
    "collect_outputs",
    # errors
    "ComfyToolkitError",
    "ComfyConnectionError",
    "ComfySubmitError",
    "ComfyExecutionError",
    "ComfyTimeoutError",
    "HistoryNotReadyError",
    "NodeNotFoundError",
    "WebSocketUnavailableError",
]
