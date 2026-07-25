"""Inject only the LoRAs a request actually needs.

Usage:
    python examples/03_dynamic_lora.py path/to/workflow_api.json

The workflow does NOT need any LoraLoader node in it. Start from a plain
checkpoint graph; the loaders are appended at run time and the downstream
model/clip edges are rewired for you.

Contrast with the usual approach of pre-wiring every LoRA at strength 0: those
loaders still merge weights, still cost load time and VRAM, and stacking many
of them visibly softens output. Injecting keeps the graph minimal.
"""

import json
import sys

from comfy_toolkit import ComfyClient, customize, find_lora_nodes

# Friendly name -> file on disk under ComfyUI/models/loras/.
# Keep this in your own config; the toolkit never ships a catalog.
LORA_CATALOG = {
    "MyStyleLoRA": "my_style_v2.safetensors",
    "DetailBoost": "detail_boost.safetensors",
    "FilmGrain": "film_grain_v1.safetensors",
}


def main(workflow_path: str) -> int:
    client = ComfyClient()

    # Whatever your UI collected. Zero-strength entries are simply not built.
    requested = {
        "MyStyleLoRA": 0.7,
        "DetailBoost": 0.35,
        "FilmGrain": 0.0,      # slider left at zero -> no node is created
    }

    workflow = customize(
        client.load_workflow(workflow_path),
        positive="a lighthouse on a cliff, overcast, cinematic",
        negative="blurry, low quality",
        seed=42,
        filename_prefix="example_lora",
        loras=requested,
        lora_catalog=LORA_CATALOG,
    )

    injected = find_lora_nodes(workflow)
    print("LoRA nodes in the submitted graph:", len(injected))
    for node_id in injected:
        inputs = workflow[node_id]["inputs"]
        print("  {0}: {1} @ {2}".format(
            node_id, inputs["lora_name"], inputs["strength_model"]))

    # Inspect the rewired graph without submitting anything:
    if "--dry-run" in sys.argv:
        print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return 0

    if not client.is_online():
        print("ComfyUI is not reachable; re-run with --dry-run to inspect.")
        return 1

    prompt_id = client.submit(workflow)
    result = client.wait(prompt_id, expect="images", max_wait=300)
    print("done:", result["filenames"])
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
