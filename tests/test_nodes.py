"""Pure-function tests for node lookup. No server required."""

import pytest

from comfy_toolkit import nodes
from comfy_toolkit.results import (
    extract_images,
    extract_videos,
    parse_history_entry,
)


# ----------------------------------------------------------------------
# Class-based lookup
# ----------------------------------------------------------------------

def test_find_nodes_by_class(simple_workflow):
    assert nodes.find_nodes_by_class(simple_workflow, "KSampler") == ["5"]
    assert nodes.find_nodes_by_class(simple_workflow, "CLIPTextEncode") == ["2", "3"]
    assert nodes.find_nodes_by_class(simple_workflow, "NopeLoader") == []


def test_find_node_by_class_returns_none_when_absent(simple_workflow):
    assert nodes.find_node_by_class(simple_workflow, "SaveImage") == "7"
    assert nodes.find_node_by_class(simple_workflow, "NopeLoader") is None


def test_find_sampler_nodes_matches_variants(simple_workflow):
    simple_workflow["99"] = {
        "inputs": {"model": ["1", 0]},
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "Refine Pass"},
    }
    assert nodes.find_sampler_nodes(simple_workflow) == ["5", "99"]


def test_iter_nodes_skips_malformed_entries(simple_workflow):
    simple_workflow["extra_metadata"] = "not a node"
    simple_workflow["also_bad"] = {"class_type": "NoInputs"}
    ids = [nid for nid, _ in nodes.iter_nodes(simple_workflow)]
    assert "extra_metadata" not in ids
    assert "also_bad" not in ids
    assert "5" in ids


# ----------------------------------------------------------------------
# Title-based lookup (bilingual)
# ----------------------------------------------------------------------

def test_find_node_by_title_is_case_insensitive(simple_workflow):
    assert nodes.find_node_by_title(simple_workflow, "positive") == "2"
    assert nodes.find_node_by_title(simple_workflow, "SAVE IMAGE") == "7"


def test_find_node_by_title_matches_chinese(simple_workflow):
    assert nodes.find_node_by_title(simple_workflow, "正向") == "2"
    assert nodes.find_node_by_title(simple_workflow, "負向") == "3"


def test_find_node_by_title_missing_returns_none(simple_workflow):
    assert nodes.find_node_by_title(simple_workflow, "no such title") is None


def test_negative_hint_wins_over_positive_hint(simple_workflow):
    # "Negative Prompt" contains "Prompt", which is a positive hint too.
    buckets = nodes.classify_text_encode_nodes(simple_workflow)
    assert buckets["positive"] == ["2"]
    assert buckets["negative"] == ["3"]


def test_classify_handles_chinese_only_titles(simple_workflow):
    simple_workflow["2"]["_meta"]["title"] = "正面提示"
    simple_workflow["3"]["_meta"]["title"] = "负面提示"
    buckets = nodes.classify_text_encode_nodes(simple_workflow)
    assert buckets["positive"] == ["2"]
    assert buckets["negative"] == ["3"]


# ----------------------------------------------------------------------
# Prompt node resolution + positional fallback
# ----------------------------------------------------------------------

def test_find_prompt_nodes_uses_titles(simple_workflow):
    found = nodes.find_prompt_nodes(simple_workflow)
    assert found == {"positive": ["2"], "negative": ["3"]}


def test_find_prompt_nodes_falls_back_to_graph_edges(untitled_workflow):
    """Default titles are ambiguous; the sampler's edges are not."""
    buckets = nodes.classify_text_encode_nodes(untitled_workflow)
    assert buckets["negative"] == []  # titles alone cannot tell them apart

    found = nodes.find_prompt_nodes(untitled_workflow)
    assert found["positive"] == ["2"]
    assert found["negative"] == ["3"]


def test_find_prompt_nodes_completes_pair_when_one_role_known(simple_workflow):
    simple_workflow["3"]["_meta"]["title"] = "Second Encoder"
    del simple_workflow["5"]["inputs"]["negative"]
    found = nodes.find_prompt_nodes(simple_workflow)
    assert found["positive"] == ["2"]
    assert found["negative"] == ["3"]


def test_find_prompt_nodes_empty_for_promptless_workflow():
    workflow = {
        "1": {"inputs": {"image": "photo.png"}, "class_type": "LoadImage",
              "_meta": {"title": "Load Image"}},
        "2": {"inputs": {"images": ["1", 0], "filename_prefix": "out"},
              "class_type": "SaveImage", "_meta": {"title": "Save Image"}},
    }
    assert nodes.find_prompt_nodes(workflow) == {"positive": [], "negative": []}


# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------

def test_text_input_key_prefers_text_and_skips_links(simple_workflow):
    assert nodes.text_input_key(simple_workflow["2"]) == "text"

    linked = {"inputs": {"text": ["42", 0]}, "class_type": "CLIPTextEncode"}
    assert nodes.text_input_key(linked) is None

    prompt_style = {"inputs": {"prompt": "hello"}, "class_type": "SomeTextEncode"}
    assert nodes.text_input_key(prompt_style) == "prompt"


def test_next_node_id(simple_workflow):
    assert nodes.next_node_id(simple_workflow) == "8"
    assert nodes.next_node_id({}) == "1"
    assert nodes.next_node_id({"3": {}, "not_numeric": {}}) == "4"


def test_find_load_image_and_lora_nodes(simple_workflow):
    assert nodes.find_load_image_nodes(simple_workflow) == []
    assert nodes.find_lora_nodes(simple_workflow) == []
    simple_workflow["8"] = {
        "inputs": {"lora_name": "example.safetensors"},
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Model Only LoRA"},
    }
    assert nodes.find_lora_nodes(simple_workflow) == ["8"]


# ----------------------------------------------------------------------
# Result extraction, including the gifs/videos gotcha
# ----------------------------------------------------------------------

IMAGE_ENTRY = {
    "status": {"status_str": "success", "completed": True},
    "outputs": {
        "7": {"images": [
            {"filename": "out_00001_.png", "subfolder": "", "type": "output"}
        ]}
    },
}

VIDEO_ENTRY = {
    "status": {"status_str": "success", "completed": True},
    "outputs": {
        "30": {"gifs": [
            {"filename": "clip_00001.mp4", "subfolder": "vids", "type": "output"}
        ]}
    },
}


def test_extract_images():
    images = extract_images(IMAGE_ENTRY)
    assert len(images) == 1
    assert images[0]["filename"] == "out_00001_.png"
    assert images[0]["node_id"] == "7"


def test_video_output_is_not_in_images_key():
    """The whole point: an mp4 arrives under 'gifs', so 'images' is empty."""
    assert extract_images(VIDEO_ENTRY) == []
    videos = extract_videos(VIDEO_ENTRY)
    assert len(videos) == 1
    assert videos[0]["filename"] == "clip_00001.mp4"
    assert videos[0]["subfolder"] == "vids"
    assert videos[0]["output_key"] == "gifs"


def test_extract_videos_reads_videos_key_too():
    entry = {"outputs": {"9": {"videos": [{"filename": "a.webm"}]}}}
    assert [v["filename"] for v in extract_videos(entry)] == ["a.webm"]


def test_extract_videos_filters_by_extension():
    entry = {"outputs": {"9": {"gifs": [
        {"filename": "preview.txt"},
        {"filename": "real.mp4"},
    ]}}}
    assert [v["filename"] for v in extract_videos(entry)] == ["real.mp4"]


def test_parse_history_entry_statuses():
    assert parse_history_entry(IMAGE_ENTRY)["status"] == "completed"
    assert parse_history_entry(VIDEO_ENTRY)["status"] == "completed"
    assert parse_history_entry({"status": {}, "outputs": {}})["status"] == "running"

    failed = parse_history_entry({"status": {"status_str": "error"}})
    assert failed["status"] == "error"
    assert failed["images"] == []


def test_parse_history_entry_collects_all_filenames():
    entry = {
        "status": {"completed": True},
        "outputs": {
            "7": {"images": [{"filename": "a.png"}]},
            "8": {"gifs": [{"filename": "b.mp4"}]},
        },
    }
    result = parse_history_entry(entry)
    assert sorted(result["filenames"]) == ["a.png", "b.mp4"]


@pytest.mark.parametrize("entry", [{}, {"outputs": {}}, {"outputs": {"1": None}}])
def test_extraction_tolerates_junk(entry):
    assert extract_images(entry) == []
    assert extract_videos(entry) == []
