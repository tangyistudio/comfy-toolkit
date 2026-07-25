"""Mutate an API-format workflow before submitting it.

The headline function here is :func:`inject_loras`, which appends brand new
``LoraLoader`` nodes to the tail of the model/clip chain and rewires every
downstream consumer.

Why not just pre-wire every LoRA at strength 0?
    That is the usual workaround: build one workflow containing all ten LoRAs
    you might ever want, leave the unused ones at ``strength 0``, and dial them
    up at runtime. It is convenient and it degrades output quality. A LoRA at
    strength 0 is not a no-op in practice: each loader still merges weights
    into the model and CLIP, and stacking many of them accumulates numerical
    drift, inflates load time and VRAM, and visibly softens results. Injecting
    only the LoRAs you actually want keeps the graph exactly as small as the
    request requires.

Because every helper here mutates the dict in place, always start from a fresh
``json.loads`` of the workflow for each submission. Then injections never
accumulate across runs.
"""

import copy
import random
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import nodes as _nodes
from .errors import NodeNotFoundError

__all__ = [
    "MODEL_INPUT_KEYS",
    "CLIP_INPUT_KEYS",
    "random_seed",
    "find_model_chain_tail",
    "inject_loras",
    "set_lora_strengths_by_title",
    "set_prompts",
    "set_seed",
    "set_filename_prefix",
    "set_input_image",
    "customize",
]

#: Input names that carry a MODEL link (output slot 0 of a loader).
MODEL_INPUT_KEYS: Tuple[str, ...] = ("model",)

#: Input names that carry a CLIP link (output slot 1 of a loader).
CLIP_INPUT_KEYS: Tuple[str, ...] = ("clip",)

_SEED_MODULUS = 1_000_000_000


def random_seed() -> int:
    """Return a fresh seed.

    Mixes the millisecond clock with a random component so that two calls
    inside the same millisecond still differ.
    """
    return (int(time.time() * 1000) + random.randint(0, 999_999)) % _SEED_MODULUS


def find_model_chain_tail(workflow: Dict) -> Optional[str]:
    """Return the node id currently at the end of the MODEL chain.

    That is whatever the sampler's ``model`` input points at — a checkpoint
    loader in a bare workflow, or the last ``LoraLoader`` in a stacked one.
    Falls back to the last LoRA loader, then to a checkpoint loader.
    """
    for nid in _nodes.find_sampler_nodes(workflow):
        link = workflow[nid]["inputs"].get("model")
        if isinstance(link, list) and link and str(link[0]) in workflow:
            return str(link[0])

    loras = _nodes.find_lora_nodes(workflow)
    if loras:
        return loras[-1]

    for class_type in ("CheckpointLoaderSimple", "CheckpointLoader",
                       "UNETLoader"):
        found = _nodes.find_node_by_class(workflow, class_type)
        if found:
            return found
    return None


def _rewire(workflow: Dict, old_tail: str, new_tail: str,
            skip: Iterable[str]) -> None:
    """Point every consumer of *old_tail* at *new_tail* instead.

    Only ``model`` and ``clip`` inputs are touched, and matching is done on
    input *name* rather than output slot number, so nodes like
    ``CLIPSetLastLayer`` or ``KSamplerAdvanced`` are rewired correctly without
    being enumerated. The new tail is always a ``LoraLoader``, whose outputs
    are fixed at MODEL=0 / CLIP=1, so the rewritten links use those slots
    regardless of how the previous tail numbered its own outputs.

    Newly created nodes are skipped so a fresh loader never feeds itself.
    """
    skip_set = set(skip)
    replacements = (
        (MODEL_INPUT_KEYS, 0),
        (CLIP_INPUT_KEYS, 1),
    )
    for nid, node in _nodes.iter_nodes(workflow):
        if nid in skip_set:
            continue
        inputs = node["inputs"]
        for keys, new_slot in replacements:
            for key in keys:
                link = inputs.get(key)
                if (isinstance(link, list) and len(link) == 2
                        and str(link[0]) == old_tail):
                    inputs[key] = [new_tail, new_slot]


def inject_loras(
    workflow: Dict,
    loras: Mapping[str, float],
    catalog: Optional[Mapping[str, str]] = None,
    *,
    model_slot: int = 0,
    clip_slot: int = 1,
    skip_zero: bool = True,
    title_prefix: str = "Injected LoRA",
) -> List[str]:
    """Append ``LoraLoader`` nodes to the model/clip chain and rewire it.

    Args:
        workflow: API-format workflow dict, mutated in place.
        loras: Mapping of LoRA key to strength, e.g. ``{"MyStyleLoRA": 0.7}``.
            Insertion order is preserved, so the chain order is yours to
            control.
        catalog: Optional mapping of LoRA key to ``.safetensors`` filename.
            When omitted, the keys in *loras* are used as filenames directly.
            Supply a catalog when you want friendly names in your UI and real
            filenames on disk.
        model_slot: Output slot of the chain tail that carries MODEL.
        clip_slot: Output slot of the chain tail that carries CLIP.
        skip_zero: Skip entries whose strength is exactly ``0.0``. This is the
            point of the whole exercise — a zero-strength LoRA should simply
            not exist in the graph.
        title_prefix: Prefix for the generated ``_meta.title``, which makes
            injected nodes obvious when you dump the workflow for debugging.

    Returns:
        The node ids that were created, in chain order. Empty if nothing was
        injected.

    Raises:
        NodeNotFoundError: The workflow has no identifiable model chain.
    """
    if not loras:
        return []

    pending: List[Tuple[str, float]] = []
    for key, strength in loras.items():
        value = float(strength)
        if skip_zero and value == 0.0:
            continue
        pending.append((key, value))
    if not pending:
        return []

    tail = find_model_chain_tail(workflow)
    if tail is None:
        raise NodeNotFoundError(
            "cannot inject LoRAs: no sampler, LoRA loader or checkpoint "
            "loader found to anchor the model chain"
        )

    created: List[str] = []
    for key, strength in pending:
        lora_name = catalog.get(key, key) if catalog else key
        node_id = _nodes.next_node_id(workflow)
        workflow[node_id] = {
            "inputs": {
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
                "model": [tail, model_slot],
                "clip": [tail, clip_slot],
            },
            "class_type": "LoraLoader",
            "_meta": {"title": "{0} - {1}".format(title_prefix, key)},
        }
        created.append(node_id)
        _rewire(workflow, tail, node_id, skip=created)
        tail = node_id
        model_slot, clip_slot = 0, 1  # LoraLoader always emits MODEL=0, CLIP=1

    return created


def set_lora_strengths_by_title(workflow: Dict,
                                strengths: Mapping[str, float]) -> List[str]:
    """Adjust strengths of LoRA loaders that are *already* in the workflow.

    Matches each key against ``_meta.title`` (case-insensitive substring), so a
    key of ``"MyStyleLoRA"`` finds a node titled ``"LoRA 3 - MyStyleLoRA"``.
    Both ``strength_model`` and ``strength_clip`` are set to the same value.

    Unmatched keys are ignored silently — a workflow is allowed to not contain
    every LoRA your UI offers.

    Returns:
        The node ids that were modified.
    """
    if not strengths:
        return []

    candidates = [(nid, (workflow[nid].get("_meta") or {}).get("title", "") or "")
                  for nid in _nodes.find_lora_nodes(workflow)]
    touched: List[str] = []
    for key, value in strengths.items():
        needle = key.lower()
        for nid, title in candidates:
            if needle in title.lower():
                inputs = workflow[nid]["inputs"]
                if "strength_model" in inputs:
                    inputs["strength_model"] = float(value)
                if "strength_clip" in inputs:
                    inputs["strength_clip"] = float(value)
                touched.append(nid)
                break
    return touched


def set_prompts(workflow: Dict,
                positive: Optional[str] = None,
                negative: Optional[str] = None,
                *,
                required: bool = True) -> Dict[str, List[str]]:
    """Write prompt text into the positive / negative encoder nodes.

    Node discovery goes through :func:`comfy_toolkit.nodes.find_prompt_nodes`,
    so bilingual titles and untitled workflows both work. The correct input
    name (``text`` or ``prompt``) is detected per node, and inputs already
    wired to another node are left alone.

    Args:
        workflow: Workflow dict, mutated in place.
        positive: Positive prompt. ``None`` leaves the workflow value as-is.
        negative: Negative prompt. ``None`` leaves the workflow value as-is.
        required: Raise if a requested role has no matching node.

    Returns:
        ``{"positive": [ids...], "negative": [ids...]}`` for the nodes written.

    Raises:
        NodeNotFoundError: *required* is true and a role could not be located.
    """
    found = _nodes.find_prompt_nodes(workflow)
    written: Dict[str, List[str]] = {"positive": [], "negative": []}

    for role, value in (("positive", positive), ("negative", negative)):
        if value is None:
            continue
        targets = found[role]
        if not targets and required:
            raise NodeNotFoundError(
                "no {0} prompt node found in workflow".format(role))
        for nid in targets:
            key = _nodes.text_input_key(workflow[nid])
            if key is not None:
                workflow[nid]["inputs"][key] = value
                written[role].append(nid)
    return written


def set_seed(workflow: Dict,
             seed: Optional[int] = None,
             *,
             stagger: int = 1000,
             all_nodes: bool = False) -> int:
    """Set the seed on sampler nodes.

    Multi-stage workflows (base pass + refine pass) get staggered seeds derived
    from a single base value: node *i* receives ``seed + i * stagger``. One
    number therefore reproduces the whole graph.

    Args:
        workflow: Workflow dict, mutated in place.
        seed: Base seed. ``None`` generates a fresh random one.
        stagger: Offset added per additional sampler. ``0`` gives every sampler
            the same seed.
        all_nodes: Also set ``seed`` on any non-sampler node that has such an
            input (upscalers, noise nodes). Those all receive the base seed.

    Returns:
        The base seed actually used, so callers can record it.
    """
    if seed is None:
        seed = random_seed()
    seed = int(seed)

    sampler_ids = _nodes.find_sampler_nodes(workflow)
    for index, nid in enumerate(sampler_ids):
        workflow[nid]["inputs"]["seed"] = (seed + index * stagger) % _SEED_MODULUS

    if all_nodes:
        for nid, node in _nodes.iter_nodes(workflow):
            if nid in sampler_ids:
                continue
            if "seed" in node["inputs"] and not isinstance(node["inputs"]["seed"], list):
                node["inputs"]["seed"] = seed
    return seed


def set_filename_prefix(workflow: Dict, prefix: str,
                        class_types: Sequence[str] = ("SaveImage",)) -> List[str]:
    """Set ``filename_prefix`` on every output-saving node.

    Defaults to ``SaveImage``; pass e.g. ``("SaveImage", "VHS_VideoCombine")``
    to also rename video output.

    Returns:
        The node ids that were modified.
    """
    touched: List[str] = []
    for nid, node in _nodes.iter_nodes(workflow):
        if node.get("class_type") in class_types and "filename_prefix" in node["inputs"]:
            node["inputs"]["filename_prefix"] = prefix
            touched.append(nid)
    return touched


def set_input_image(workflow: Dict, filename: str,
                    title_hint: Optional[str] = None,
                    *,
                    index: int = 0) -> str:
    """Point a ``LoadImage`` node at *filename* (relative to ComfyUI's input dir).

    Args:
        workflow: Workflow dict, mutated in place.
        filename: File name already present in ComfyUI's ``input/`` folder.
        title_hint: Prefer the ``LoadImage`` whose title contains this keyword.
            Useful for multi-reference workflows where slots are titled.
        index: Positional fallback when *title_hint* is absent or unmatched.

    Returns:
        The node id that was modified.

    Raises:
        NodeNotFoundError: No suitable ``LoadImage`` node exists.
    """
    load_ids = _nodes.find_load_image_nodes(workflow)
    if not load_ids:
        raise NodeNotFoundError("workflow has no LoadImage node")

    target = None
    if title_hint:
        needle = title_hint.lower()
        for nid in load_ids:
            title = (workflow[nid].get("_meta") or {}).get("title", "") or ""
            if needle in title.lower():
                target = nid
                break
    if target is None:
        if index >= len(load_ids):
            raise NodeNotFoundError(
                "workflow has {0} LoadImage node(s); index {1} is out of "
                "range".format(len(load_ids), index)
            )
        target = load_ids[index]

    workflow[target]["inputs"]["image"] = filename
    return target


def customize(
    workflow: Dict,
    *,
    positive: Optional[str] = None,
    negative: Optional[str] = None,
    seed: Optional[int] = None,
    filename_prefix: Optional[str] = None,
    loras: Optional[Mapping[str, float]] = None,
    lora_catalog: Optional[Mapping[str, str]] = None,
    existing_lora_strengths: Optional[Mapping[str, float]] = None,
    input_image: Optional[str] = None,
    copy_workflow: bool = True,
    required_prompts: bool = False,
) -> Dict:
    """Apply the usual per-submission edits in one call.

    Order of operations: prompts, existing LoRA strengths, injected LoRAs,
    input image, seed, filename prefix.

    Args:
        copy_workflow: Deep-copy first and return the copy, leaving the caller's
            template untouched. Set to ``False`` to mutate in place.
        required_prompts: Raise when a prompt was supplied but no matching node
            exists. Off by default so image-only workflows (upscale, face
            restore) can go through the same helper.

    Returns:
        The customized workflow dict.
    """
    work = copy.deepcopy(workflow) if copy_workflow else workflow

    if positive is not None or negative is not None:
        set_prompts(work, positive, negative, required=required_prompts)
    if existing_lora_strengths:
        set_lora_strengths_by_title(work, existing_lora_strengths)
    if loras:
        inject_loras(work, loras, lora_catalog)
    if input_image is not None:
        set_input_image(work, input_image)
    set_seed(work, seed)
    if filename_prefix is not None:
        set_filename_prefix(work, filename_prefix)
    return work
