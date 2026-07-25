"""Tolerant node lookup helpers for ComfyUI API-format workflows.

An API-format workflow is a plain ``dict`` mapping node id (a string) to::

    {
        "inputs": {...},
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"}
    }

Every function in this module is pure: it reads a workflow dict and returns
node ids or plain data. Nothing here touches the network, so all of it is
unit-testable against a fixture JSON file.

Why title-based lookup?
    Hardcoding node ids (``wf["6"]["inputs"]["text"] = ...``) breaks the moment
    a user re-saves the workflow from the ComfyUI canvas. Looking a node up by
    its ``_meta.title`` survives re-saves, and titles are what humans actually
    edit. Titles are matched case-insensitively against both English and
    Chinese keywords, because bilingual workflow authors are common.

Why the structural fallback?
    Titles are optional. When a workflow has untouched default titles, the
    graph itself still says which encoder is positive: ``KSampler.inputs
    ["positive"]`` points straight at it. :func:`find_prompt_nodes` uses titles
    first and falls back to following those edges.

    Provenance, since the two halves are not equally battle-tested: the
    bilingual title matching was carried over from production code, where
    workflows authored in mixed English/Chinese made it necessary. The
    edge-following fallback was designed for this package to cover workflows
    it does not control, and is backed by unit tests rather than by production
    mileage.
"""

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "POSITIVE_TITLE_HINTS",
    "NEGATIVE_TITLE_HINTS",
    "SAMPLER_CLASS_HINTS",
    "TEXT_INPUT_KEYS",
    "iter_nodes",
    "find_nodes_by_class",
    "find_node_by_class",
    "find_nodes_by_class_contains",
    "find_nodes_by_title",
    "find_node_by_title",
    "find_sampler_nodes",
    "find_text_encode_nodes",
    "classify_text_encode_nodes",
    "find_prompt_nodes",
    "find_load_image_nodes",
    "find_lora_nodes",
    "text_input_key",
    "next_node_id",
]

#: Title keywords that mark a text-encode node as the positive prompt.
#: Both English and Chinese (traditional + simplified) spellings are accepted.
POSITIVE_TITLE_HINTS: Tuple[str, ...] = (
    "positive",
    "prompt",
    "正向",
    "正面",
    "正",
)

#: Title keywords that mark a text-encode node as the negative prompt.
#: Checked *before* the positive hints, since "Negative Prompt" contains
#: "Prompt" and would otherwise be misfiled.
NEGATIVE_TITLE_HINTS: Tuple[str, ...] = (
    "negative",
    "負向",
    "负向",
    "負面",
    "负面",
    "負",
    "负",
)

#: Substrings identifying sampler-like node classes. Matched case-sensitively
#: against ``class_type`` with ``in``, so ``KSampler``, ``KSamplerAdvanced``
#: and ``SamplerCustom`` all qualify.
SAMPLER_CLASS_HINTS: Tuple[str, ...] = ("KSampler", "SamplerCustom")

#: Input names that carry prompt text, in preference order.
#:
#: There is no single convention, which is why this is a list and not a
#: constant. Vanilla ``CLIPTextEncode`` — and everything modelled on it —
#: names the field ``text``. The image-editing encoders that ship with the
#: Qwen-Image-Edit node pack (``TextEncodeQwenImageEdit``,
#: ``TextEncodeQwenImageEditPlus``) name it ``prompt`` instead. Code that
#: writes ``inputs["text"]`` unconditionally silently does nothing on the
#: second family: the node keeps whatever prompt was baked into the exported
#: JSON, the job succeeds, and the output has nothing to do with the request.
#:
#: :func:`text_input_key` therefore picks per node, by inspecting which key is
#: actually present, rather than by matching ``class_type`` against a list
#: that would go stale with every new node pack.
TEXT_INPUT_KEYS: Tuple[str, ...] = ("text", "prompt")


def iter_nodes(workflow: Dict):
    """Yield ``(node_id, node)`` for every well-formed node in *workflow*.

    Malformed entries (non-dict values) are skipped rather than raising, so a
    workflow carrying stray metadata keys still works.
    """
    for node_id, node in workflow.items():
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
            yield str(node_id), node


def _title(node: Dict) -> str:
    meta = node.get("_meta")
    if isinstance(meta, dict):
        title = meta.get("title")
        if isinstance(title, str):
            return title
    return ""


def find_nodes_by_class(workflow: Dict, class_type: str) -> List[str]:
    """Return every node id whose ``class_type`` equals *class_type* exactly."""
    return [nid for nid, node in iter_nodes(workflow)
            if node.get("class_type") == class_type]


def find_node_by_class(workflow: Dict, class_type: str) -> Optional[str]:
    """Return the first node id of *class_type*, or ``None``."""
    found = find_nodes_by_class(workflow, class_type)
    return found[0] if found else None


def find_nodes_by_class_contains(workflow: Dict,
                                 needles: Sequence[str]) -> List[str]:
    """Return node ids whose ``class_type`` contains any of *needles*."""
    out = []
    for nid, node in iter_nodes(workflow):
        class_type = node.get("class_type") or ""
        if any(needle in class_type for needle in needles):
            out.append(nid)
    return out


def find_nodes_by_title(workflow: Dict, keyword: str) -> List[str]:
    """Return node ids whose ``_meta.title`` contains *keyword*.

    Matching is case-insensitive and substring-based, so the keyword
    ``"upscale"`` finds a node titled ``"Stage 2 - Upscale (optional)"``.
    """
    needle = keyword.lower()
    return [nid for nid, node in iter_nodes(workflow)
            if needle in _title(node).lower()]


def find_node_by_title(workflow: Dict, keyword: str) -> Optional[str]:
    """Return the first node id whose title contains *keyword*, or ``None``."""
    found = find_nodes_by_title(workflow, keyword)
    return found[0] if found else None


def find_sampler_nodes(workflow: Dict) -> List[str]:
    """Return every sampler-like node id (``KSampler``, ``KSamplerAdvanced``...)."""
    return find_nodes_by_class_contains(workflow, SAMPLER_CLASS_HINTS)


def find_text_encode_nodes(workflow: Dict) -> List[str]:
    """Return every text-encoding node id.

    Covers ``CLIPTextEncode`` and the various ``*TextEncode*`` variants shipped
    by custom node packs, without needing to enumerate them.
    """
    return find_nodes_by_class_contains(workflow, ("TextEncode",))


def classify_text_encode_nodes(workflow: Dict) -> Dict[str, List[str]]:
    """Split text-encode nodes into positive / negative / unknown by title.

    Returns a dict with keys ``"positive"``, ``"negative"`` and ``"unknown"``,
    each mapping to a list of node ids in workflow order.

    Negative hints win over positive hints, because a title such as
    ``"Negative Prompt"`` matches both.
    """
    buckets: Dict[str, List[str]] = {
        "positive": [], "negative": [], "unknown": []}
    for nid in find_text_encode_nodes(workflow):
        title = _title(workflow[nid]).lower()
        if any(hint.lower() in title for hint in NEGATIVE_TITLE_HINTS):
            buckets["negative"].append(nid)
        elif any(hint.lower() in title for hint in POSITIVE_TITLE_HINTS):
            buckets["positive"].append(nid)
        else:
            buckets["unknown"].append(nid)
    return buckets


def _prompt_nodes_from_sampler(workflow: Dict) -> Tuple[List[str], List[str]]:
    """Follow ``sampler.inputs.positive / .negative`` edges to their sources."""
    positive: List[str] = []
    negative: List[str] = []
    for nid in find_sampler_nodes(workflow):
        inputs = workflow[nid]["inputs"]
        for role, bucket in (("positive", positive), ("negative", negative)):
            link = inputs.get(role)
            if isinstance(link, list) and link:
                src = str(link[0])
                if src in workflow and src not in bucket:
                    bucket.append(src)
    return positive, negative


def find_prompt_nodes(workflow: Dict) -> Dict[str, List[str]]:
    """Locate the positive and negative prompt nodes.

    Strategy, in order:

    1. ``_meta.title`` keywords (bilingual, see :data:`POSITIVE_TITLE_HINTS`),
       accepted only when they actually separate the two roles.
    2. Positional fallback: follow the sampler's ``positive`` / ``negative``
       input edges back to whatever node feeds them. This rescues workflows
       whose encoders both carry ComfyUI's default title.
    3. If exactly two text-encode nodes exist and only one role was resolved,
       the remaining node takes the other role.

    Returns:
        ``{"positive": [ids...], "negative": [ids...]}``. Either list may be
        empty if the workflow genuinely has no such node.
    """
    buckets = classify_text_encode_nodes(workflow)
    positive = list(buckets["positive"])
    negative = list(buckets["negative"])

    # 1. Titles resolved both roles unambiguously.
    if positive and negative and not set(positive) & set(negative):
        return {"positive": positive, "negative": negative}

    # 2. Structural truth: the sampler says which conditioning is which.
    edge_pos, edge_neg = _prompt_nodes_from_sampler(workflow)
    if edge_pos and edge_neg and not set(edge_pos) & set(edge_neg):
        return {"positive": edge_pos, "negative": edge_neg}

    positive = positive or edge_pos
    negative = negative or edge_neg

    # 3. Two encoders, one role known -> the other encoder takes the rest.
    encoders = find_text_encode_nodes(workflow)
    if len(encoders) == 2:
        if positive and not negative:
            negative = [n for n in encoders if n not in positive]
        elif negative and not positive:
            positive = [n for n in encoders if n not in negative]

    return {"positive": positive, "negative": negative}


def find_load_image_nodes(workflow: Dict) -> List[str]:
    """Return every ``LoadImage`` node id."""
    return find_nodes_by_class(workflow, "LoadImage")


def find_lora_nodes(workflow: Dict) -> List[str]:
    """Return every LoRA loader node id (``LoraLoader``, model-only variants)."""
    return [nid for nid, node in iter_nodes(workflow)
            if "lora" in (node.get("class_type") or "").lower()]


def text_input_key(node: Dict) -> Optional[str]:
    """Return the input name holding prompt text for *node*, or ``None``.

    Skips inputs that are wired to another node (represented as a
    ``[node_id, slot]`` list) — overwriting those would silently break the
    graph.
    """
    inputs = node.get("inputs") or {}
    for key in TEXT_INPUT_KEYS:
        if key in inputs and not isinstance(inputs[key], list):
            return key
    return None


def next_node_id(workflow: Dict) -> str:
    """Return an unused numeric node id, as a string.

    ComfyUI ids are strings but conventionally numeric. Non-numeric ids in the
    workflow are ignored when computing the maximum.
    """
    numeric = [int(k) for k in workflow.keys() if str(k).lstrip("-").isdigit()]
    return str((max(numeric) + 1) if numeric else 1)
