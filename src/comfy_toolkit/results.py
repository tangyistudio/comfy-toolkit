"""Read outputs out of a ComfyUI ``/history`` entry.

The one thing everybody gets wrong: **video nodes do not write to ``images``.**
``SaveImage`` puts its files under ``outputs[node_id]["images"]``, but
``VHS_VideoCombine`` and friends put theirs under ``"gifs"`` — and, depending
on the node pack version, sometimes ``"videos"``. Code that only reads
``"images"`` sees an empty output list and concludes the job failed, even
though the mp4 is sitting on disk.

:func:`extract_videos` reads both keys and filters by file extension, so it
keeps working when a node pack renames its output bucket.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

__all__ = [
    "IMAGE_OUTPUT_KEYS",
    "VIDEO_OUTPUT_KEYS",
    "VIDEO_EXTENSIONS",
    "extract_images",
    "extract_videos",
    "entry_status",
    "parse_history_entry",
    "output_url_params",
    "write_sidecar",
    "collect_outputs",
]

#: ``outputs`` keys that hold still images.
IMAGE_OUTPUT_KEYS: Sequence[str] = ("images",)

#: ``outputs`` keys that hold animations / video. ``gifs`` is what
#: ``VHS_VideoCombine`` actually uses, whatever the file extension is.
VIDEO_OUTPUT_KEYS: Sequence[str] = ("gifs", "videos")

#: Extensions accepted as video-ish output.
VIDEO_EXTENSIONS: Sequence[str] = (".mp4", ".webm", ".gif", ".mkv", ".mov", ".avi")


def _files_from(entry: Dict, keys: Sequence[str],
                extensions: Optional[Sequence[str]] = None) -> List[Dict]:
    """Flatten ``entry["outputs"][*][key]`` into a list of file descriptors."""
    found: List[Dict] = []
    outputs = entry.get("outputs") or {}
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for key in keys:
            for item in node_output.get(key) or []:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename") or ""
                if extensions and not filename.lower().endswith(tuple(extensions)):
                    continue
                found.append({
                    "node_id": str(node_id),
                    "filename": filename,
                    "subfolder": item.get("subfolder", "") or "",
                    "type": item.get("type", "output") or "output",
                    "output_key": key,
                })
    return found


def extract_images(entry: Dict) -> List[Dict]:
    """Return image file descriptors from a history entry.

    Each descriptor is ``{"node_id", "filename", "subfolder", "type",
    "output_key"}`` — the same fields ComfyUI's ``/view`` endpoint expects.
    """
    return _files_from(entry, IMAGE_OUTPUT_KEYS)


def extract_videos(entry: Dict) -> List[Dict]:
    """Return video / animation file descriptors from a history entry.

    Reads both ``gifs`` and ``videos`` buckets and keeps anything with a known
    video extension, so an mp4 emitted under the ``gifs`` key is found.
    """
    return _files_from(entry, VIDEO_OUTPUT_KEYS, VIDEO_EXTENSIONS)


def entry_status(entry: Dict) -> str:
    """Return ``"error"``, ``"completed"`` or ``"running"`` for a history entry.

    Note that ``"completed"`` here only means ComfyUI marked execution done; it
    does not promise the outputs list is non-empty.
    """
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        return "error"
    if status.get("completed"):
        return "completed"
    return "running"


def parse_history_entry(entry: Dict) -> Dict:
    """Normalize a history entry into a single result dict.

    Returns:
        ``{"status": ..., "images": [...], "videos": [...], "filenames": [...]}``
        where status is one of ``"completed"``, ``"running"`` or ``"error"``.
        An entry with any output file is reported as ``"completed"`` even if
        the status block has not caught up yet.
    """
    status = entry_status(entry)
    if status == "error":
        return {
            "status": "error",
            "error": str(entry.get("status") or {}),
            "images": [],
            "videos": [],
            "filenames": [],
        }

    images = extract_images(entry)
    videos = extract_videos(entry)
    if images or videos:
        status = "completed"

    return {
        "status": status,
        "images": images,
        "videos": videos,
        "filenames": [f["filename"] for f in images + videos],
    }


def output_url_params(descriptor: Dict) -> Dict[str, str]:
    """Build the query parameters for ComfyUI's ``/view`` endpoint.

    Pass the result to :meth:`comfy_toolkit.ComfyClient.download` to fetch the
    bytes without touching ComfyUI's output folder directly — which matters
    when the server is on another machine.
    """
    return {
        "filename": descriptor.get("filename", ""),
        "subfolder": descriptor.get("subfolder", ""),
        "type": descriptor.get("type", "output"),
    }


def write_sidecar(path: Union[str, Path], meta: Dict) -> Path:
    """Write *meta* as pretty-printed UTF-8 JSON next to a generated file.

    Args:
        path: Destination ``.json`` path, or the media file whose suffix should
            be replaced with ``.json``.
        meta: Any JSON-serializable mapping.

    Returns:
        The path written.
    """
    target = Path(path)
    if target.suffix.lower() != ".json":
        target = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def collect_outputs(
    descriptors: Sequence[Dict],
    source_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    meta: Optional[Dict] = None,
    *,
    sidecars: bool = True,
    batch_meta_name: str = "meta.json",
) -> List[Path]:
    """Copy generated files out of ComfyUI's output folder into your own tree.

    Only usable when ComfyUI runs on the same filesystem. For a remote server,
    download via :func:`output_url_params` instead.

    Args:
        descriptors: File descriptors from :func:`extract_images` /
            :func:`extract_videos`.
        source_dir: ComfyUI's ``output`` directory.
        dest_dir: Where to put the copies; created if missing.
        meta: Optional metadata. When given, a batch-level ``meta.json`` is
            written and (if *sidecars*) a per-file ``<name>.json`` too.
        sidecars: Write one JSON sidecar per copied file.
        batch_meta_name: Filename for the batch-level metadata.

    Returns:
        Paths of the files successfully copied. Missing sources are skipped
        rather than raising, so one absent file cannot lose the whole batch.
    """
    source_root = Path(source_dir)
    dest_root = Path(dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for index, descriptor in enumerate(descriptors):
        filename = descriptor.get("filename") or ""
        if not filename:
            continue
        subfolder = descriptor.get("subfolder") or ""
        src = source_root / subfolder / filename if subfolder else source_root / filename
        if not src.exists():
            continue
        dst = dest_root / Path(filename).name
        shutil.copy2(src, dst)
        copied.append(dst)

        if meta is not None and sidecars:
            item_meta = dict(meta)
            item_meta["filename"] = dst.name
            item_meta["index"] = index
            write_sidecar(dst, item_meta)

    if meta is not None:
        write_sidecar(dest_root / batch_meta_name, meta)

    return copied
