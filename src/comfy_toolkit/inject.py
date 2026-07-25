"""Mutate an API-format workflow before submitting it.

The headline function here is :func:`inject_loras`, which appends brand new
``LoraLoader`` nodes to the tail of the model/clip chain and rewires every
downstream consumer.

Why not just pre-wire every LoRA at strength 0?
    That is the usual workaround: build one workflow containing all ten LoRAs
    you might ever want, leave the unused ones at ``strength 0``, and dial them
    up at runtime.

    To be accurate about it: strength 0 really is a no-op in ComfyUI.
    ``LoraLoader.load_lora()`` begins with ``if strength_model == 0 and
    strength_clip == 0: return (model, clip)``, and that early return happens
    before the file is opened — no VRAM, no load time, no effect on the image.
    Pre-wiring is not a performance problem.

    It is a workflow-management problem. Pre-wiring forces the LoRA list to be
    decided when the graph is drawn, so every change means re-exporting the API
    JSON; it pushes callers into hardcoding node ids, which break the next time
    the workflow is re-saved; it cannot express a catalog that is only known at
    runtime; and it does not work at all on a workflow the *user* brought.
    Injecting instead means any plain checkpoint graph can accept exactly the
    LoRAs one request asked for, with no prior edits.

Mutation and copying
    The individual helpers in this module mutate the workflow dict in place and
    return the ids they touched. :func:`customize` is the exception: it
    deep-copies first by default (``copy_workflow=True``) and returns the copy,
    so a template can be reused across submissions without injections piling
    up. Pass ``copy_workflow=False`` to opt into in-place behaviour. If you call
    the helpers directly, either copy the workflow yourself or re-read it from
    disk for each submission.
"""

import copy
import random
import time
from typing import (
    Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from . import nodes as _nodes
from .errors import NodeNotFoundError

__all__ = [
    "MODEL_INPUT_KEYS",
    "CLIP_INPUT_KEYS",
    "MODEL_ONLY_CLASS_HINTS",
    "BATCH_SEED_SPACING",
    "random_seed",
    "batch_seeds",
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

#: ``class_type`` substrings marking a loader that emits MODEL but no CLIP.
#: In such graphs CLIP comes from a separate ``CLIPLoader``, so a LoRA appended
#: to the model chain must not try to read a CLIP output that does not exist.
#:
#: **Heuristic, and deliberately a fallback.** ComfyUI has no registry of node
#: output types that a client can consult offline, so this is a hand-maintained
#: list of class-name substrings, not derived knowledge. It is consulted only
#: when :func:`_tail_emits_clip` cannot answer the question structurally, and
#: an unrecognised class is assumed to emit CLIP. Custom node packs invent new
#: loader names constantly; extend this tuple if yours is not covered.
MODEL_ONLY_CLASS_HINTS: Tuple[str, ...] = (
    "UNETLoader",
    "UnetLoader",
    "DiffusionModelLoader",
    "ModelOnly",
)

#: ComfyUI's model-only LoRA loader, used when the chain tail has no CLIP.
_MODEL_ONLY_LORA_CLASS = "LoraLoaderModelOnly"

_SEED_MODULUS = 1_000_000_000

#: Default gap between the seeds of consecutive items in a batch.
#:
#: 7919 is prime. Two properties follow, and they are the whole reason for the
#: number: a prime spacing shares no factor with the round numbers people reach
#: for elsewhere (1000 for :func:`set_seed`'s multi-pass stagger, 10/100 for a
#: hand-written loop), so a batch's seeds cannot land on top of another
#: sequence's; and it is large enough that the per-item bands never collide
#: with the within-workflow stagger of a multi-sampler graph.
BATCH_SEED_SPACING = 7919


def random_seed() -> int:
    """Return a fresh seed.

    Mixes the millisecond clock with a random component so that two calls
    inside the same millisecond still differ.
    """
    return (int(time.time() * 1000) + random.randint(0, 999_999)) % _SEED_MODULUS


def batch_seeds(count: int,
                base: Optional[int] = None,
                *,
                spacing: int = BATCH_SEED_SPACING) -> List[int]:
    """Return *count* well-spread seeds for a batch of submissions.

    Submitting N variations of one prompt needs N seeds that are reproducible
    from a single recorded number and far enough apart to stay distinct. Item
    *i* gets ``base + i * spacing``.

    ``base + i`` (spacing 1) is the tempting version and it is the one to
    avoid: seeds one apart run into each other as soon as anything else in the
    pipeline also offsets a seed — most immediately :func:`set_seed`, which
    staggers multi-pass samplers *within* one workflow. With the default
    spacing, each batch item owns a band ~7919 wide, which no realistic
    within-workflow stagger can walk out of.

    Args:
        count: How many seeds to produce. Must be >= 1.
        base: First seed. ``None`` draws a fresh one via :func:`random_seed`,
            which is the number to log if you want the batch back later.
        spacing: Gap between consecutive seeds. See
            :data:`BATCH_SEED_SPACING`.

    Returns:
        A list of *count* seeds, ``base`` first.

    Raises:
        ValueError: *count* is less than 1.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if base is None:
        base = random_seed()
    base = int(base)
    return [(base + i * int(spacing)) % _SEED_MODULUS for i in range(count)]


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


def _tail_emits_clip(workflow: Dict, tail: str) -> bool:
    """Return whether the node *tail* has a CLIP output to hang a LoRA off.

    Decided structurally first: if anything in the graph already reads ``clip``
    from *tail*, it plainly has one. Otherwise fall back to the class name, so
    that ``UNETLoader``-style model-only loaders (whose CLIP arrives from a
    separate ``CLIPLoader``) are recognised even in a graph where no CLIP
    consumer has been wired yet.

    Unknown classes default to ``True``, which preserves the behaviour for
    ordinary checkpoint loaders.
    """
    for _, node in _nodes.iter_nodes(workflow):
        for key in CLIP_INPUT_KEYS:
            link = node["inputs"].get(key)
            if (isinstance(link, list) and len(link) == 2
                    and str(link[0]) == tail):
                return True

    class_type = (workflow.get(tail) or {}).get("class_type") or ""
    return not any(hint in class_type for hint in MODEL_ONLY_CLASS_HINTS)


def _rewire(workflow: Dict, old_tail: str, new_tail: str,
            skip: Iterable[str], *, clip: bool = True) -> None:
    """Point every consumer of *old_tail* at *new_tail* instead.

    Only ``model`` and ``clip`` inputs are touched, and matching is done on
    input *name* rather than output slot number, so nodes like
    ``CLIPSetLastLayer`` or ``KSamplerAdvanced`` are rewired correctly without
    being enumerated. The new tail is always a LoRA loader, whose outputs are
    fixed at MODEL=0 / CLIP=1, so the rewritten links use those slots
    regardless of how the previous tail numbered its own outputs.

    Args:
        clip: Rewire ``clip`` inputs too. ``False`` for a model-only loader,
            which has no CLIP output to point them at.

    Newly created nodes are skipped so a fresh loader never feeds itself.
    """
    skip_set = set(skip)
    replacements = [(MODEL_INPUT_KEYS, 0)]
    if clip:
        replacements.append((CLIP_INPUT_KEYS, 1))
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
            filenames on disk. A catalog scanned from ``models/loras/`` can run
            to hundreds of entries, so log a count and a handful of names
            rather than dumping the mapping.
        model_slot: Output slot of the chain tail that carries MODEL. The
            default suits ``CheckpointLoaderSimple`` and any ``LoraLoader``
            already in the chain.
        clip_slot: Output slot of the chain tail that carries CLIP. Ignored
            when the tail emits no CLIP at all — see below.
        skip_zero: Skip entries whose strength is exactly ``0.0``, so the
            submitted graph contains only the loaders that were actually asked
            for. This is about keeping the graph small and readable, not about
            speed: ComfyUI's ``LoraLoader`` already early-returns on a
            zero-strength pair without reading the file.
        title_prefix: Prefix for the generated ``_meta.title``, which makes
            injected nodes obvious when you dump the workflow for debugging.

    Model-only chains:
        If the chain tail has no CLIP output — ``UNETLoader`` and similar, where
        CLIP is supplied by a separate ``CLIPLoader`` — a
        ``LoraLoaderModelOnly`` node is created instead of a ``LoraLoader``, no
        ``clip`` link is written, and downstream ``clip`` inputs are left
        pointing wherever they already pointed.

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

        # A tail like UNETLoader emits MODEL only; writing a clip link would
        # point at an output slot that does not exist. Use ComfyUI's
        # model-only loader instead and leave CLIP alone.
        with_clip = _tail_emits_clip(workflow, tail)
        if with_clip:
            inputs: Dict = {
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
                "model": [tail, model_slot],
                "clip": [tail, clip_slot],
            }
            class_type = "LoraLoader"
        else:
            inputs = {
                "lora_name": lora_name,
                "strength_model": strength,
                "model": [tail, model_slot],
            }
            class_type = _MODEL_ONLY_LORA_CLASS

        workflow[node_id] = {
            "inputs": inputs,
            "class_type": class_type,
            "_meta": {"title": "{0} - {1}".format(title_prefix, key)},
        }
        created.append(node_id)
        _rewire(workflow, tail, node_id, skip=created, clip=with_clip)
        tail = node_id
        # Both LoRA loaders emit MODEL=0; only LoraLoader also emits CLIP=1.
        model_slot, clip_slot = 0, 1

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


def _load_image_haystack(node: Dict) -> str:
    """Lowercased title + currently configured filename of a ``LoadImage``.

    Both are searched because a fixed reference slot is identifiable by either
    one: a careful workflow author titles it, and a careless one still leaves
    the anchor's filename sitting in the ``image`` field of the exported JSON.
    """
    title = (node.get("_meta") or {}).get("title", "") or ""
    current = node.get("inputs", {}).get("image", "")
    if isinstance(current, list):  # wired to another node, not a filename
        current = ""
    return "{0}\n{1}".format(title, current).lower()


def set_input_image(workflow: Dict, filename: str,
                    title_hint: Optional[str] = None,
                    *,
                    index: int = 0,
                    skip_titles: Optional[Iterable[str]] = None,
                    skip_predicate: Optional[Callable[[str, Dict], bool]] = None,
                    ) -> str:
    """Point a ``LoadImage`` node at *filename* (relative to ComfyUI's input dir).

    Multiple ``LoadImage`` nodes, one of which must not be touched:
        Plenty of real workflows carry a second ``LoadImage`` holding a *fixed*
        reference image — a face or style anchor, an IPAdapter reference, a
        control map — that is meant to stay exactly where it is while the
        per-request image goes into the other slot. "Take the first
        ``LoadImage``" is a coin flip there, and losing the flip silently
        overwrites the anchor: the job still succeeds, and the output is
        quietly wrong.

        *skip_titles* is the cheap way out. Give it the substrings that mark
        your fixed slots and they are removed from consideration before any
        other rule runs. Matching is case-insensitive and looks at both the
        node's ``_meta.title`` and the filename already sitting in its
        ``image`` field, since an anchor is usually recognisable from one or
        the other. *skip_predicate* covers anything hairier.

    Args:
        workflow: Workflow dict, mutated in place.
        filename: File name already present in ComfyUI's ``input/`` folder.
        title_hint: Prefer the ``LoadImage`` whose title contains this keyword.
            Useful for multi-reference workflows where slots are titled.
        index: Positional fallback when *title_hint* is absent or unmatched.
            Indexes the *surviving* candidates, not the raw node list.
        skip_titles: Substrings marking nodes to exclude. See above.
        skip_predicate: Called as ``skip_predicate(node_id, node)``; a truthy
            return excludes that node. Applied after *skip_titles*.

    Returns:
        The node id that was modified.

    Raises:
        NodeNotFoundError: No suitable ``LoadImage`` node exists, or every
            candidate was excluded. Excluding everything raises rather than
            falling back to an excluded node: the caller said those slots hold
            something that must survive, and writing to one anyway would
            reintroduce exactly the silent corruption the argument exists to
            prevent.
    """
    load_ids = _nodes.find_load_image_nodes(workflow)
    if not load_ids:
        raise NodeNotFoundError("workflow has no LoadImage node")

    candidates = list(load_ids)
    if skip_titles:
        needles = [str(s).lower() for s in skip_titles if str(s)]
        if needles:
            candidates = [
                nid for nid in candidates
                if not any(n in _load_image_haystack(workflow[nid])
                           for n in needles)
            ]
    if skip_predicate is not None:
        candidates = [nid for nid in candidates
                      if not skip_predicate(nid, workflow[nid])]

    if not candidates:
        raise NodeNotFoundError(
            "every one of the {0} LoadImage node(s) was excluded by "
            "skip_titles / skip_predicate; nothing left to write to".format(
                len(load_ids))
        )

    target = None
    if title_hint:
        needle = title_hint.lower()
        for nid in candidates:
            title = (workflow[nid].get("_meta") or {}).get("title", "") or ""
            if needle in title.lower():
                target = nid
                break
    if target is None:
        if index >= len(candidates):
            raise NodeNotFoundError(
                "workflow has {0} selectable LoadImage node(s); index {1} is "
                "out of range".format(len(candidates), index)
            )
        target = candidates[index]

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
    input_image_title_hint: Optional[str] = None,
    input_image_skip_titles: Optional[Iterable[str]] = None,
    copy_workflow: bool = True,
    required_prompts: bool = False,
) -> Dict:
    """Apply the usual per-submission edits in one call.

    Order of operations: prompts, existing LoRA strengths, injected LoRAs,
    input image, seed, filename prefix.

    Args:
        input_image_title_hint: Passed to :func:`set_input_image` as
            *title_hint*.
        input_image_skip_titles: Passed to :func:`set_input_image` as
            *skip_titles* — the escape hatch for workflows whose second
            ``LoadImage`` is a fixed reference slot that must not be
            overwritten.
        copy_workflow: Deep-copy first and return the copy, leaving the caller's
            template untouched. Set to ``False`` to mutate in place. Either way
            the workflow you hand to a *second* submission must be clean:
            injections are cumulative, so reuse the untouched template (or
            re-read it from disk), never the return value of a previous call.
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
        set_input_image(work, input_image,
                        input_image_title_hint,
                        skip_titles=input_image_skip_titles)
    set_seed(work, seed)
    if filename_prefix is not None:
        set_filename_prefix(work, filename_prefix)
    return work
