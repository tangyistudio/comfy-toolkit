"""Tests for workflow mutation: LoRA injection, prompts, seeds, prefixes.

Everything here runs against the fixture workflow in-process. No server.
"""

import pytest

from comfy_toolkit import inject, nodes
from comfy_toolkit.errors import NodeNotFoundError


# ----------------------------------------------------------------------
# Chain tail discovery
# ----------------------------------------------------------------------

def test_find_model_chain_tail_is_checkpoint_when_bare(simple_workflow):
    assert inject.find_model_chain_tail(simple_workflow) == "1"


def test_find_model_chain_tail_follows_existing_lora(simple_workflow):
    inject.inject_loras(simple_workflow, {"FirstLoRA": 0.5})
    assert inject.find_model_chain_tail(simple_workflow) == "8"


def test_find_model_chain_tail_without_sampler():
    workflow = {
        "1": {"inputs": {"ckpt_name": "example_checkpoint.safetensors"},
              "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load"}},
    }
    assert inject.find_model_chain_tail(workflow) == "1"


# ----------------------------------------------------------------------
# LoRA injection
# ----------------------------------------------------------------------

def test_inject_single_lora_creates_node_and_rewires(simple_workflow):
    created = inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.7})
    assert created == ["8"]

    node = simple_workflow["8"]
    assert node["class_type"] == "LoraLoader"
    assert node["inputs"]["lora_name"] == "MyStyleLoRA"
    assert node["inputs"]["strength_model"] == 0.7
    assert node["inputs"]["strength_clip"] == 0.7
    # New loader hangs off the old tail...
    assert node["inputs"]["model"] == ["1", 0]
    assert node["inputs"]["clip"] == ["1", 1]
    # ...and every downstream consumer now reads from it.
    assert simple_workflow["5"]["inputs"]["model"] == ["8", 0]
    assert simple_workflow["2"]["inputs"]["clip"] == ["8", 1]
    assert simple_workflow["3"]["inputs"]["clip"] == ["8", 1]


def test_inject_does_not_touch_unrelated_links(simple_workflow):
    inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.7})
    # VAEDecode still reads the VAE straight from the checkpoint.
    assert simple_workflow["6"]["inputs"]["vae"] == ["1", 2]
    assert simple_workflow["5"]["inputs"]["positive"] == ["2", 0]
    assert simple_workflow["5"]["inputs"]["latent_image"] == ["4", 0]


def test_inject_multiple_loras_builds_a_chain(simple_workflow):
    created = inject.inject_loras(
        simple_workflow, {"MyStyleLoRA": 0.7, "DetailLoRA": 0.4})
    assert created == ["8", "9"]

    assert simple_workflow["8"]["inputs"]["model"] == ["1", 0]
    assert simple_workflow["9"]["inputs"]["model"] == ["8", 0]
    assert simple_workflow["9"]["inputs"]["clip"] == ["8", 1]
    # Only the last link in the chain reaches the sampler.
    assert simple_workflow["5"]["inputs"]["model"] == ["9", 0]
    assert simple_workflow["2"]["inputs"]["clip"] == ["9", 1]


def test_injected_node_never_feeds_itself(simple_workflow):
    inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.7})
    assert simple_workflow["8"]["inputs"]["model"] != ["8", 0]
    assert simple_workflow["8"]["inputs"]["clip"] != ["8", 1]


def test_zero_strength_loras_are_not_injected_at_all(simple_workflow):
    """The reason this package exists: a 0-strength LoRA should not be loaded."""
    created = inject.inject_loras(
        simple_workflow, {"MyStyleLoRA": 0.0, "DetailLoRA": 0.6})
    assert created == ["8"]
    assert simple_workflow["8"]["inputs"]["lora_name"] == "DetailLoRA"
    assert len(nodes.find_lora_nodes(simple_workflow)) == 1


def test_all_zero_means_untouched_workflow(simple_workflow):
    before = simple_workflow["5"]["inputs"]["model"]
    assert inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.0}) == []
    assert nodes.find_lora_nodes(simple_workflow) == []
    assert simple_workflow["5"]["inputs"]["model"] == before


def test_skip_zero_can_be_disabled(simple_workflow):
    created = inject.inject_loras(
        simple_workflow, {"MyStyleLoRA": 0.0}, skip_zero=False)
    assert created == ["8"]
    assert simple_workflow["8"]["inputs"]["strength_model"] == 0.0


def test_catalog_maps_friendly_names_to_filenames(simple_workflow):
    catalog = {"MyStyleLoRA": "my_style_v2.safetensors"}
    inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.8}, catalog)
    assert simple_workflow["8"]["inputs"]["lora_name"] == "my_style_v2.safetensors"
    assert "MyStyleLoRA" in simple_workflow["8"]["_meta"]["title"]


def test_catalog_miss_falls_back_to_the_key(simple_workflow):
    inject.inject_loras(
        simple_workflow, {"other.safetensors": 0.5}, {"Known": "known.safetensors"})
    assert simple_workflow["8"]["inputs"]["lora_name"] == "other.safetensors"


def test_empty_mapping_is_a_no_op(simple_workflow):
    assert inject.inject_loras(simple_workflow, {}) == []


def test_inject_raises_without_a_model_chain():
    workflow = {
        "1": {"inputs": {"images": ["2", 0], "filename_prefix": "out"},
              "class_type": "SaveImage", "_meta": {"title": "Save"}},
    }
    with pytest.raises(NodeNotFoundError):
        inject.inject_loras(workflow, {"MyStyleLoRA": 0.7})


def test_injection_does_not_accumulate_on_a_fresh_copy(simple_workflow):
    """Reload/copy per submission and injections never stack up."""
    first = inject.customize(simple_workflow, loras={"MyStyleLoRA": 0.7})
    second = inject.customize(simple_workflow, loras={"MyStyleLoRA": 0.7})
    assert len(nodes.find_lora_nodes(first)) == 1
    assert len(nodes.find_lora_nodes(second)) == 1
    assert nodes.find_lora_nodes(simple_workflow) == []


# ----------------------------------------------------------------------
# Adjusting pre-wired LoRAs
# ----------------------------------------------------------------------

def test_set_lora_strengths_by_title(simple_workflow):
    inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.7, "DetailLoRA": 0.4})
    touched = inject.set_lora_strengths_by_title(
        simple_workflow, {"DetailLoRA": 0.9})
    assert touched == ["9"]
    assert simple_workflow["9"]["inputs"]["strength_model"] == 0.9
    assert simple_workflow["9"]["inputs"]["strength_clip"] == 0.9
    assert simple_workflow["8"]["inputs"]["strength_model"] == 0.7


def test_set_lora_strengths_ignores_unknown_keys(simple_workflow):
    inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.7})
    assert inject.set_lora_strengths_by_title(
        simple_workflow, {"NotPresent": 0.5}) == []


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

def test_set_prompts_writes_both_roles(simple_workflow):
    written = inject.set_prompts(
        simple_workflow, "a lighthouse at dawn", "blurry, jpeg artifacts")
    assert written == {"positive": ["2"], "negative": ["3"]}
    assert simple_workflow["2"]["inputs"]["text"] == "a lighthouse at dawn"
    assert simple_workflow["3"]["inputs"]["text"] == "blurry, jpeg artifacts"


def test_set_prompts_leaves_none_alone(simple_workflow):
    original = simple_workflow["3"]["inputs"]["text"]
    inject.set_prompts(simple_workflow, positive="new positive")
    assert simple_workflow["2"]["inputs"]["text"] == "new positive"
    assert simple_workflow["3"]["inputs"]["text"] == original


def test_set_prompts_works_on_untitled_workflow(untitled_workflow):
    inject.set_prompts(untitled_workflow, "sunrise", "noise")
    assert untitled_workflow["2"]["inputs"]["text"] == "sunrise"
    assert untitled_workflow["3"]["inputs"]["text"] == "noise"


def test_set_prompts_skips_wired_text_inputs(simple_workflow):
    simple_workflow["2"]["inputs"]["text"] = ["99", 0]
    written = inject.set_prompts(simple_workflow, positive="ignored")
    assert written["positive"] == []
    assert simple_workflow["2"]["inputs"]["text"] == ["99", 0]


def test_set_prompts_required_raises_when_missing():
    workflow = {
        "1": {"inputs": {"image": "photo.png"}, "class_type": "LoadImage",
              "_meta": {"title": "Load Image"}},
    }
    with pytest.raises(NodeNotFoundError):
        inject.set_prompts(workflow, positive="anything", required=True)
    # Not required -> silently does nothing.
    assert inject.set_prompts(workflow, positive="anything", required=False) == {
        "positive": [], "negative": []}


# ----------------------------------------------------------------------
# Seeds, prefixes, input images
# ----------------------------------------------------------------------

def test_set_seed_explicit(simple_workflow):
    assert inject.set_seed(simple_workflow, 12345) == 12345
    assert simple_workflow["5"]["inputs"]["seed"] == 12345


def test_set_seed_random_is_deterministic_in_return(simple_workflow):
    used = inject.set_seed(simple_workflow)
    assert simple_workflow["5"]["inputs"]["seed"] == used


def test_set_seed_staggers_multiple_samplers(simple_workflow):
    simple_workflow["99"] = {
        "inputs": {"seed": 0, "model": ["1", 0]},
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "Refine"},
    }
    inject.set_seed(simple_workflow, 1000, stagger=7)
    assert simple_workflow["5"]["inputs"]["seed"] == 1000
    assert simple_workflow["99"]["inputs"]["seed"] == 1007


def test_set_seed_all_nodes(simple_workflow):
    simple_workflow["99"] = {
        "inputs": {"seed": 0}, "class_type": "SomeUpscaler",
        "_meta": {"title": "Upscale"},
    }
    inject.set_seed(simple_workflow, 555, all_nodes=True)
    assert simple_workflow["99"]["inputs"]["seed"] == 555


def test_set_filename_prefix(simple_workflow):
    assert inject.set_filename_prefix(simple_workflow, "run_001") == ["7"]
    assert simple_workflow["7"]["inputs"]["filename_prefix"] == "run_001"


def test_set_filename_prefix_for_video_nodes(simple_workflow):
    simple_workflow["20"] = {
        "inputs": {"filename_prefix": "old", "images": ["6", 0]},
        "class_type": "VHS_VideoCombine",
        "_meta": {"title": "Video Combine"},
    }
    touched = inject.set_filename_prefix(
        simple_workflow, "clip_001", class_types=("SaveImage", "VHS_VideoCombine"))
    assert sorted(touched) == ["20", "7"]


def test_set_input_image_by_title_and_index():
    workflow = {
        "1": {"inputs": {"image": "a.png"}, "class_type": "LoadImage",
              "_meta": {"title": "Slot 1 - Subject"}},
        "2": {"inputs": {"image": "b.png"}, "class_type": "LoadImage",
              "_meta": {"title": "Slot 2 - Reference"}},
    }
    assert inject.set_input_image(workflow, "new.png", "Reference") == "2"
    assert workflow["2"]["inputs"]["image"] == "new.png"
    assert inject.set_input_image(workflow, "first.png", index=0) == "1"
    assert workflow["1"]["inputs"]["image"] == "first.png"


def test_set_input_image_errors(simple_workflow):
    with pytest.raises(NodeNotFoundError):
        inject.set_input_image(simple_workflow, "x.png")


# ----------------------------------------------------------------------
# customize()
# ----------------------------------------------------------------------

def test_customize_applies_everything_at_once(simple_workflow):
    result = inject.customize(
        simple_workflow,
        positive="a lighthouse at dawn",
        negative="blurry",
        seed=42,
        filename_prefix="demo",
        loras={"MyStyleLoRA": 0.7},
        lora_catalog={"MyStyleLoRA": "my_style_v2.safetensors"},
    )
    assert result["2"]["inputs"]["text"] == "a lighthouse at dawn"
    assert result["3"]["inputs"]["text"] == "blurry"
    assert result["5"]["inputs"]["seed"] == 42
    assert result["7"]["inputs"]["filename_prefix"] == "demo"
    assert result["8"]["inputs"]["lora_name"] == "my_style_v2.safetensors"
    assert result["5"]["inputs"]["model"] == ["8", 0]
    # Template untouched.
    assert "8" not in simple_workflow


def test_customize_in_place(simple_workflow):
    result = inject.customize(
        simple_workflow, positive="x", seed=7, copy_workflow=False)
    assert result is simple_workflow
    assert simple_workflow["2"]["inputs"]["text"] == "x"
