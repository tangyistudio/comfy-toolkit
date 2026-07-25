# comfy-toolkit

A small Python client for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)'s HTTP API — **zero dependencies** by default, stdlib `urllib` only.

It exists to handle three specific things that any project automating ComfyUI has to deal with sooner or later, and that are easy to get subtly wrong.

---

## 1. The `/history` race condition

ComfyUI's WebSocket announces completion **before** `/history/{prompt_id}` is guaranteed to be readable. Your job succeeded, the file is on disk, and your code sees nothing. There is a window between "the executor is done" and "the history entry has been written", and the client has no way to know in advance how wide it will be — so the failure is intermittent rather than reproducible.

The `Before` / `After` pairs below sketch the shape of each problem and this package's answer to it. They are illustrations, not a changelog of this repository.

**Before** — treats the socket message as the result:

```python
ws.recv()                                  # "execution_success"
history = requests.get(f"{url}/history/{pid}").json()
images = history[pid]["outputs"]["9"]["images"]   # KeyError, intermittently
```

A fixed `sleep()` here is the usual patch, and it is the wrong shape: too short and it still races, too long and every generation pays for the worst case.

**After** — the socket is a hint, never the result:

```python
def on_progress(value, maximum):          # the callback signature is (value, max)
    print("\r{0}/{1}".format(value, maximum), end="", flush=True)

result = client.submit_and_track(workflow, on_progress=on_progress)
print(result["filenames"])   # guaranteed non-empty, or a typed exception
```

`on_progress` is called as `on_progress(value, maximum)` — both ints, `maximum > 0`. Exceptions raised inside it are swallowed so a broken progress bar cannot kill a generation, which also means a callback with the wrong signature fails silently and the bar simply never moves. Write it as a named two-argument function and it will be obvious if it doesn't fit.

`submit_and_track()` re-reads `/history` up to 20 times (~20s budget) after the socket says done, accepting only an entry that actually contains output files. Retrying costs nothing on the happy path — the first read normally succeeds and nothing is waited for — and the budget is only spent when something is genuinely wrong. If every retry comes back empty you get `HistoryNotReadyError`: a real failure, not a guess. Raise `history_retries` if your setup wants a longer grace period.

## 2. Dynamic LoRA injection, without pre-wiring

The common workaround is to build one mega-workflow containing all ten LoRAs you might ever want, leave the unused ones at **strength 0**, and dial them up at run time.

First, what this is *not*. A strength-0 LoRA in ComfyUI is a genuine no-op — `LoraLoader.load_lora()` opens with:

```python
if strength_model == 0 and strength_clip == 0:
    return (model, clip)
```

That early return happens *before* the file is read, so an unused loader costs no VRAM, no load time, and changes nothing about the image. Claims that a strength-0 LoRA still merges weights or softens output are simply not true of ComfyUI. So this is not a performance argument, and this package will not pretend otherwise.

It's a workflow-management problem, and that's a real one:

- **The LoRA list has to be decided at design time.** Every LoRA a request might want must already be wired into the graph, so the graph carries permanent clutter for options most runs never use.
- **Changing the list means re-exporting.** Add one LoRA and you're back in the canvas re-doing *Save (API Format)*, then updating whatever code indexes into it.
- **Driving it from code means hardcoding node ids.** `wf["14"]["inputs"]["strength_model"] = 0.7` breaks silently the next time someone re-saves the workflow and the ids shift.
- **The catalog is often only known at run time** — LoRAs your app loads from config, or whatever the user happens to have in `models/loras/`. You cannot pre-wire against a list you don't have yet.
- **It only works on workflows you control.** A user who brings their own plain checkpoint graph would have to edit it first.

**Before** — the list is baked into the graph, the ids are baked into the code:

```python
for node_id, key in LORA_NODE_IDS.items():        # hardcoded ids, breaks on re-save
    wf[node_id]["inputs"]["strength_model"] = strengths.get(key, 0.0)
```

**After** — the graph starts clean and gets exactly the loaders this request asked for:

```python
workflow = customize(workflow, loras={"MyStyleLoRA": 0.7, "FilmGrain": 0.0})
# one LoraLoader node created; FilmGrain at 0.0 is skipped, so the graph stays minimal
```

`inject_loras()` appends fresh `LoraLoader` nodes at the tail of the model chain and rewires every downstream `model` / `clip` edge. Your workflow file needs **no** LoRA nodes at all — start from a plain checkpoint graph, including one a user drew themselves.

Zero-strength entries are skipped (`skip_zero=True`) to keep the submitted graph readable, not to save cycles — ComfyUI would have ignored them anyway.

A catalog built by scanning `models/loras/` routinely runs to hundreds of entries. When you log it, or surface it in an error message, print the count and the first few names — the full list is unreadable and buries whatever you were actually debugging.

### ⚠️ Load the workflow fresh for every submission

The single most common way to break dynamic injection. Injection **appends** nodes; it does not reconcile them against what a previous call left behind. Reuse the same dict for a second submission and the second batch of `LoraLoader` nodes lands on top of the first, the chain gets rewired twice, and the graph you submit is not the graph you think you built. It usually still runs, which is what makes it expensive to find.

```python
template = client.load_workflow("my_workflow_api.json")   # read once

for prompt in prompts:
    wf = customize(template, positive=prompt, loras={"MyStyleLoRA": 0.7})
    client.submit(wf)            # `template` is never mutated
```

`customize()` deep-copies by default (`copy_workflow=True`) and returns the copy, so the pattern above is safe as written — the rule is only that the thing you pass *in* must always be the pristine template, never the return value of a previous call. Two ways to get it wrong:

```python
wf = customize(wf, ...)                      # feeding the result back in: accumulates
wf = customize(template, ..., copy_workflow=False)   # mutates `template` itself
```

The helpers below `customize()` — `inject_loras()`, `set_prompts()` and friends — all mutate in place by design. Call those directly and copying, or re-reading from disk, is on you.

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

Not on PyPI yet (release pending) — install from git:

```bash
# core, zero dependencies
pip install git+https://github.com/tangyistudio/comfy-toolkit

# + websocket-client, for live progress via submit_and_track()
pip install "comfy-toolkit[ws] @ git+https://github.com/tangyistudio/comfy-toolkit"
```

Once it is published, `pip install comfy-toolkit` / `pip install "comfy-toolkit[ws]"` will work as usual.

## Quickstart

`submit_and_track()` is the one entry point that needs the optional `[ws]` extra; everything else in this example is stdlib-only.

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
result = client.submit_and_track(workflow, on_progress=lambda done, total: print(done, "/", total))
print(result["filenames"])                              # needs the [ws] extra
```

Server address resolution: `ComfyClient(base_url=...)` → `COMFY_URL` env var → `http://127.0.0.1:8188`.

## API reference

### `ComfyClient`

| Method | Purpose |
| --- | --- |
| `ComfyClient(base_url=None, client_id=None, timeout=10.0)` | Construct; falls back to `COMFY_URL`, then localhost |
| `.is_online()` | `True` if the server answers |
| `.system_stats()` / `.vram_stats(device_index=0)` | Raw stats / VRAM totals in bytes **and** GB |
| `.queue_status()` | `{"running": N, "pending": N}`; an unreachable server or an unparseable reply reads as an empty queue instead of raising |
| `.load_workflow(path)` | Read an API-format workflow JSON |
| `.submit(workflow)` | Queue it, return `prompt_id` |
| `.submit_and_track(workflow, **kw)` | Submit + WebSocket progress + race fix. Needs the `[ws]` extra. **Defaults to `expect="images"` — pass `expect="videos"` for video workflows**, or it will raise `HistoryNotReadyError` on a job that succeeded |
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
| `inject_loras(wf, loras, catalog=None, *, model_slot=0, clip_slot=1, skip_zero=True)` | Append `LoraLoader` nodes and rewire the chain |
| `set_lora_strengths_by_title(wf, strengths)` | Adjust LoRAs already present, matched by title |
| `set_prompts(wf, positive, negative)` | Write prompt text; picks the `text` / `prompt` input per node — see below |
| `set_seed(wf, seed=None, stagger=1000)` | One base seed, staggered across multi-pass samplers *within* one workflow |
| `batch_seeds(count, base=None, spacing=7919)` | Well-spread seeds *across* the submissions of a batch — see below |
| `set_filename_prefix(wf, prefix, class_types=...)` | Rename outputs (`SaveImage`, `VHS_VideoCombine`, ...) |
| `set_input_image(wf, filename, title_hint=None, *, index=0, skip_titles=None, skip_predicate=None)` | Point a `LoadImage` at a file, excluding fixed reference slots — see below |
| `find_model_chain_tail(wf)` | The node currently feeding the sampler's `model` |

**`model_slot` / `clip_slot`, and model-only chains.** A LoRA loader has to be told which output slots of the *current* chain tail carry MODEL and CLIP. The defaults (`0` and `1`) are right for `CheckpointLoaderSimple` and for any `LoraLoader` already in the chain; override them if your tail is an exotic loader that numbers its outputs differently. Some tails emit no CLIP at all — `UNETLoader` and friends, where CLIP comes from a separate `CLIPLoader`. `inject_loras()` detects that case, emits `LoraLoaderModelOnly` instead of `LoraLoader`, and leaves `clip_slot` unused rather than writing a link to an output slot that does not exist.

That detection is structural first: if anything in the graph already reads `clip` from the tail, it plainly has one. Only when nothing does — a graph where no CLIP consumer has been wired yet — does it fall back to `MODEL_ONLY_CLASS_HINTS`, a **hand-maintained list of class-name substrings**. Treat that list as a heuristic, not knowledge: ComfyUI exposes no offline registry of node output types, custom packs invent loader names continuously, and an unrecognised class is assumed to emit CLIP. If your loader is not covered, extend the tuple.

### Multiple `LoadImage` nodes, one of which is a fixed reference

A workflow with two `LoadImage` nodes usually is not offering you a choice. One takes the per-request image; the other holds a **fixed reference image — a face or style anchor, an IPAdapter reference, a control map** — that is supposed to stay exactly where it is. "Write to the first `LoadImage`" is a coin flip, and losing it overwrites the anchor. Nothing errors. The job succeeds. The output is just quietly wrong, and it looks like a model problem rather than a plumbing one.

```python
set_input_image(
    workflow,
    "portrait_042.png",
    skip_titles=["style anchor", "ref_"],   # matched against title AND current filename
)
```

Matching is case-insensitive and checks both `_meta.title` and the filename already sitting in the node's `image` field, because an anchor slot is usually recognisable from one or the other — a careful author titles it, and a careless one still leaves the anchor's filename in the exported JSON. `skip_predicate(node_id, node)` handles anything more complicated. `index` then counts across the *surviving* candidates.

If every candidate is excluded, `set_input_image()` raises `NodeNotFoundError` rather than falling back to an excluded node. You said those slots hold something that must survive; writing to one anyway would recreate exactly the silent corruption the argument exists to prevent.

`customize()` forwards `input_image_title_hint` and `input_image_skip_titles` for the same purpose.

### Seeds: `stagger` within a workflow, `batch_seeds()` across submissions

Two different axes, and they are easy to conflate.

`set_seed(wf, seed, stagger=1000)` handles *one* workflow that contains several samplers — a base pass plus a refine pass — giving sampler *i* `seed + i * 1000` so a single recorded number reproduces the whole graph.

`batch_seeds()` handles the other axis: N submissions of the *same* workflow that must differ from each other.

```python
from comfy_toolkit import batch_seeds, customize

base = None                                     # or a number you want to reproduce
for i, seed in enumerate(batch_seeds(8, base)):
    wf = customize(template, positive=prompt, seed=seed,
                   filename_prefix="run_{0:03d}".format(i))
    client.submit(wf)
```

The default spacing is **7919, a prime**. `base + i` is the tempting version and the one to avoid: seeds one apart collide as soon as anything else in the pipeline also offsets a seed — most immediately `set_seed()`'s within-workflow stagger. A prime spacing shares no factor with the round numbers used elsewhere (1000, 10, 100), so a batch's seeds cannot land on a sequence some other part of the pipeline already generated, and 7919 is wide enough that each batch item owns a band no realistic multi-sampler stagger can walk out of.

Pass `base` explicitly to reproduce a batch; leave it `None` and log the first returned seed.

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

**Why title lookup.** Hardcoding `wf["6"]["inputs"]["text"]` breaks the moment a user re-saves the workflow from the canvas. Titles survive re-saves, and `find_prompt_nodes()` matches them in **English and Chinese** (`Positive` / `Negative` / `正向` / `負向` / `正面` / `负面`), with negative hints taking priority since "Negative Prompt" also contains "Prompt". This half came over from production code, where mixed-language workflow titles made it necessary.

**Why the structural fallback.** If titles are ambiguous — both encoders left at ComfyUI's default `CLIP Text Encode (Prompt)` — `find_prompt_nodes()` falls back to the graph itself and follows `KSampler.inputs.positive` / `.negative` back to their sources, so user-drawn workflows work too. To be clear about provenance: this half is **new in this package**, designed for workflows it does not control. It is covered by unit tests, not by production mileage.

**`text` vs `prompt`: the same field under two names.** There is no single convention for what a text encoder calls its text input, and getting it wrong fails silently.

| Node class | Input name |
| --- | --- |
| `CLIPTextEncode` (and everything modelled on it) | `text` |
| `TextEncodeQwenImageEdit`, `TextEncodeQwenImageEditPlus` | `prompt` |

Write `inputs["text"]` unconditionally and on the second family nothing happens at all: the node keeps whatever prompt was baked into the exported JSON, the job succeeds, and the image has nothing to do with what was asked for. `set_prompts()` resolves the key per node by looking at which one is actually present (`TEXT_INPUT_KEYS`, `text` first), rather than matching `class_type` against a list that goes stale with every new node pack. Inputs already wired to another node are left alone — overwriting a link with a string would break the graph.

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
- `websocket-client` only if you want live progress — the `[ws]` extra, or just `pip install websocket-client`

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

Built by [Tangyi Studio](https://github.com/tangyistudio).
