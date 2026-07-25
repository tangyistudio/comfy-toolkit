# comfy-toolkit

A small Python client for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)'s HTTP API — **zero dependencies** by default, stdlib `urllib` only.

It exists because three specific things bite everyone who automates ComfyUI, and every project ends up re-solving them badly.

---

## 1. The `/history` race condition

ComfyUI's WebSocket announces completion **before** `/history/{prompt_id}` is guaranteed to be readable. Your job succeeded, the file is on disk, and your code sees nothing. It fails intermittently — worse on large images, video encodes, slow disks, and network shares. This is why so many ComfyUI integrations carry a mysterious `sleep(2)`.

**Before** — works on your laptop, fails in production:

```python
ws.recv()                                  # "execution_success"
history = requests.get(f"{url}/history/{pid}").json()
images = history[pid]["outputs"]["9"]["images"]   # KeyError, intermittently
```

**After** — the socket is a hint, never the result:

```python
result = client.submit_and_track(workflow, on_progress=bar.update)
print(result["filenames"])   # guaranteed non-empty, or a typed exception
```

`submit_and_track()` re-reads `/history` up to 20 times (~20s budget) after the socket says done, accepting only an entry that actually contains output files. A fixed sleep can't be tuned correctly: too short and it still races, too long and every generation pays for the worst case. Retrying costs nothing on the happy path. If every retry comes back empty you get `HistoryNotReadyError` — a real failure, not a guess.

## 2. Dynamic LoRA injection, without pre-wiring

The common workaround is to build one mega-workflow containing all ten LoRAs you might want, leave the unused ones at **strength 0**, and dial them up at run time. It's convenient and it degrades quality: a strength-0 LoRA is not a no-op — each loader still merges weights into the model and CLIP, accumulates numerical drift, inflates load time and VRAM, and visibly softens results.

**Before** — every LoRA is always loaded:

```python
for node_id, key in LORA_NODE_IDS.items():        # hardcoded ids, breaks on re-save
    wf[node_id]["inputs"]["strength_model"] = strengths.get(key, 0.0)
```

**After** — only the LoRAs you asked for exist in the graph:

```python
workflow = customize(workflow, loras={"MyStyleLoRA": 0.7, "FilmGrain": 0.0})
# one LoraLoader node created; FilmGrain at 0.0 is never built
```

`inject_loras()` appends fresh `LoraLoader` nodes at the tail of the model chain and rewires every downstream `model` / `clip` edge. Your workflow file needs **no** LoRA nodes at all — start from a plain checkpoint graph.

## 3. Video outputs land in `gifs`, not `images`

`SaveImage` writes to `outputs[node]["images"]`. `VHS_VideoCombine` writes to `outputs[node]["gifs"]` — yes, even for mp4 — and some node-pack versions use `"videos"`. Code that only reads `"images"` concludes the job produced nothing.

**Before:**

```python
files = entry["outputs"][node]["images"]   # empty; the mp4 is under "gifs"
```

**After:**

```python
videos = client.wait_for_videos(prompt_id)   # reads gifs + videos, filters by extension
```

---

## Install

```bash
pip install comfy-toolkit           # core, zero dependencies
pip install "comfy-toolkit[ws]"     # + websocket-client, for live progress
```

## Quickstart

```python
from comfy_toolkit import ComfyClient, customize

client = ComfyClient()                                  # or COMFY_URL env var
workflow = client.load_workflow("my_workflow_api.json") # "Save (API Format)" export
workflow = customize(
    workflow,
    positive="a quiet harbour at sunrise, soft light",
    negative="blurry, low quality",
    loras={"MyStyleLoRA": 0.7},
    seed=42,
)
result = client.submit_and_track(workflow, on_progress=lambda s, t: print(s, "/", t))
print(result["filenames"])
```

Server address resolution: `ComfyClient(base_url=...)` → `COMFY_URL` env var → `http://127.0.0.1:8188`.

## API reference

### `ComfyClient`

| Method | Purpose |
| --- | --- |
| `ComfyClient(base_url=None, client_id=None, timeout=10.0)` | Construct; falls back to `COMFY_URL`, then localhost |
| `.is_online()` | `True` if the server answers |
| `.system_stats()` / `.vram_stats(device_index=0)` | Raw stats / VRAM totals in bytes **and** GB |
| `.queue_status()` | `{"running": N, "pending": N}`; never raises |
| `.load_workflow(path)` | Read an API-format workflow JSON |
| `.submit(workflow)` | Queue it, return `prompt_id` |
| `.submit_and_track(workflow, **kw)` | Submit + WebSocket progress + race fix |
| `.history(prompt_id)` | Raw history entry, or `None` |
| `.result(prompt_id)` | Normalized `{status, images, videos, filenames}` |
| `.wait(prompt_id, expect="any")` | Poll until output exists |
| `.wait_for_images(...)` / `.wait_for_videos(...)` | Typed shortcuts (video defaults to a 30 min budget) |
| `.download(descriptor)` / `.save(descriptor, dest)` | Fetch bytes via `/view` — works with a remote server |
| `.interrupt()` | Cancel the running job |

### Workflow mutation — `comfy_toolkit.inject`

| Function | Purpose |
| --- | --- |
| `customize(wf, ...)` | Prompts + LoRAs + seed + prefix in one call; deep-copies by default |
| `inject_loras(wf, loras, catalog=None)` | Append `LoraLoader` nodes and rewire the chain |
| `set_lora_strengths_by_title(wf, strengths)` | Adjust LoRAs already present, matched by title |
| `set_prompts(wf, positive, negative)` | Write prompt text; detects `text` vs `prompt` input |
| `set_seed(wf, seed=None, stagger=1000)` | One base seed, staggered across multi-pass samplers |
| `set_filename_prefix(wf, prefix, class_types=...)` | Rename outputs (`SaveImage`, `VHS_VideoCombine`, ...) |
| `set_input_image(wf, filename, title_hint=None)` | Point a `LoadImage` at a file, by title or position |
| `find_model_chain_tail(wf)` | The node currently feeding the sampler's `model` |

### Node lookup — `comfy_toolkit.nodes`

Every function is pure and unit-tested against a fixture; no server needed.

| Function | Purpose |
| --- | --- |
| `find_nodes_by_class(wf, class_type)` | Exact `class_type` match |
| `find_node_by_title(wf, keyword)` | Case-insensitive `_meta.title` substring |
| `find_prompt_nodes(wf)` | Positive/negative encoders — see below |
| `find_sampler_nodes(wf)` | `KSampler`, `KSamplerAdvanced`, `SamplerCustom`, ... |
| `find_text_encode_nodes(wf)` / `find_lora_nodes(wf)` / `find_load_image_nodes(wf)` | Category lookups |
| `next_node_id(wf)` | An unused numeric id |

**Why title lookup, and why the fallback.** Hardcoding `wf["6"]["inputs"]["text"]` breaks the moment a user re-saves the workflow from the canvas. Titles survive re-saves, and `find_prompt_nodes()` matches them in **English and Chinese** (`Positive` / `Negative` / `正向` / `負向` / `正面` / `负面`), with negative hints taking priority since "Negative Prompt" also contains "Prompt". If titles are ambiguous — both encoders left at ComfyUI's default `CLIP Text Encode (Prompt)` — it falls back to the graph itself and follows `KSampler.inputs.positive` / `.negative` to their sources. User-drawn workflows just work.

### Results — `comfy_toolkit.results`

| Function | Purpose |
| --- | --- |
| `extract_images(entry)` / `extract_videos(entry)` | Pull file descriptors out of a history entry |
| `parse_history_entry(entry)` | Normalize to `{status, images, videos, filenames}` |
| `output_url_params(descriptor)` | Query params for `/view` |
| `write_sidecar(path, meta)` | Per-file JSON metadata next to the output |
| `collect_outputs(descriptors, src, dst, meta)` | Copy into your own library tree (same filesystem) |

### Errors — `comfy_toolkit.errors`

All derive from `ComfyToolkitError`: `ComfyConnectionError`, `ComfySubmitError`, `ComfyExecutionError`, `ComfyTimeoutError`, `HistoryNotReadyError`, `NodeNotFoundError`, `WebSocketUnavailableError`.

## Requirements

- Python 3.9+
- A running ComfyUI server, reachable over HTTP
- A workflow exported via **Save (API Format)** — the canvas `.json` is a different, incompatible schema
- `websocket-client` only if you want live progress (`pip install "comfy-toolkit[ws]"`)

## Examples

- `examples/01_basic_submit.py` — submit and wait
- `examples/02_progress_tracking.py` — live progress bar, race handled
- `examples/03_dynamic_lora.py` — inject LoRAs into a plain checkpoint workflow (`--dry-run` prints the rewired graph)

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite is entirely offline — node lookup and injection are pure functions over a fixture workflow.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Tangyi Studio](https://github.com/TangyiStudio).
