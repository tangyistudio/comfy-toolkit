"""Inject only the LoRAs a request actually needs.

Usage:
    python examples/03_dynamic_lora.py path/to/workflow_api.json
    python examples/03_dynamic_lora.py --dry-run path/to/workflow_api.json

The workflow does NOT need any LoraLoader node in it. Start from a plain
checkpoint graph; the loaders are appended at run time and the downstream
model/clip edges are rewired for you.

Contrast with the usual approach of pre-wiring every LoRA at strength 0. That
is not a quality problem — ComfyUI's LoraLoader early-returns on strength 0
without ever reading the file, so an unused loader costs nothing. It is a
workflow problem: pre-wiring fixes the LoRA list when the graph is drawn, so
this catalog could not live in code, adding an entry would mean re-exporting
the API JSON, and the code driving it would have to hardcode node ids that
break on the next re-save. Injection keeps the graph clean and the list here.
"""

import argparse
import json

from comfy_toolkit import ComfyClient, customize, find_lora_nodes

# Friendly name -> file on disk under ComfyUI/models/loras/.
# Keep this in your own config; the toolkit never ships a catalog.
LORA_CATALOG = {
    "MyStyleLoRA": "my_style_v2.safetensors",
    "DetailBoost": "detail_boost.safetensors",
    "FilmGrain": "film_grain_v1.safetensors",
}


def main(workflow_path: str, dry_run: bool = False) -> int:
    client = ComfyClient()

    # Whatever your UI collected. Zero-strength entries are simply not built,
    # so the submitted graph shows exactly what this request asked for.
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
    if dry_run:
        print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return 0

    if not client.is_online():
        print("ComfyUI is not reachable; re-run with --dry-run to inspect.")
        return 1

    prompt_id = client.submit(workflow)
    result = client.wait(prompt_id, expect="images", max_wait=300)
    print("done:", result["filenames"])
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject LoRAs into a plain checkpoint workflow.")
    parser.add_argument("workflow",
                        help="path to an API-format workflow JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rewired graph instead of submitting it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args.workflow, dry_run=args.dry_run))
