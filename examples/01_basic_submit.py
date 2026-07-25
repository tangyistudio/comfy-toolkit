"""Submit a workflow and wait for the images.

Usage:
    python examples/01_basic_submit.py path/to/workflow_api.json

Export the workflow from ComfyUI with "Save (API Format)".
Set COMFY_URL if your server is not on http://127.0.0.1:8188.
"""

import sys

from comfy_toolkit import ComfyClient, customize


def main(workflow_path: str) -> int:
    client = ComfyClient()  # base_url arg > COMFY_URL env > 127.0.0.1:8188
    print("server:", client.base_url)

    if not client.is_online():
        print("ComfyUI is not reachable. Start it, or set COMFY_URL.")
        return 1

    vram = client.vram_stats()
    print("device: {0} ({1} GB free)".format(vram["name"], vram["vram_free_gb"]))
    print("queue:", client.queue_status())

    workflow = client.load_workflow(workflow_path)
    workflow = customize(
        workflow,
        positive="a quiet harbour at sunrise, soft light, wide shot",
        negative="blurry, low quality, watermark",
        seed=None,                 # None -> a fresh random seed
        filename_prefix="example_basic",
    )

    prompt_id = client.submit(workflow)
    print("submitted:", prompt_id)

    result = client.wait(prompt_id, expect="images", max_wait=300)
    for image in result["images"]:
        print("  ->", image["filename"])
        # Fetch the bytes through the server, so this works remotely too:
        # client.save(image, "./downloads")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
