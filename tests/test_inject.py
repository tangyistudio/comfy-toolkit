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
    """A 0-strength entry produces no node, so the graph stays minimal.

    ComfyUI would ignore such a loader anyway (LoraLoader early-returns on
    strength 0), so this is about keeping the submitted graph readable rather
    than about saving any work on the server.
    """
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


def test_inject_on_model_only_chain_writes_no_clip_link():
    """A UNETLoader tail has no CLIP output, so no clip link may be written."""
    workflow = {
        "1": {"inputs": {"unet_name": "flux.safetensors"},
              "class_type": "UNETLoader", "_meta": {"title": "Load Diffusion"}},
        "2": {"inputs": {"clip_name": "t5.safetensors"},
              "class_type": "CLIPLoader", "_meta": {"title": "Load CLIP"}},
        "3": {"inputs": {"text": "hi", "clip": ["2", 0]},
              "class_type": "CLIPTextEncode", "_meta": {"title": "Positive"}},
        "4": {"inputs": {"model": ["1", 0], "positive": ["3", 0], "seed": 1},
              "class_type": "KSampler", "_meta": {"title": "KSampler"}},
    }
    created = inject.inject_loras(workflow, {"MyStyleLoRA": 0.7})
    assert created == ["5"]

    node = workflow["5"]
    assert node["class_type"] == "LoraLoaderModelOnly"
    assert "clip" not in node["inputs"]
    assert "strength_clip" not in node["inputs"]
    assert node["inputs"]["model"] == ["1", 0]
    # The model edge is rewired; the CLIP edge still comes from the CLIPLoader.
    assert workflow["4"]["inputs"]["model"] == ["5", 0]
    assert workflow["3"]["inputs"]["clip"] == ["2", 0]


def test_model_only_chain_stays_model_only_across_multiple_loras():
    workflow = {
        "1": {"inputs": {"unet_name": "flux.safetensors"},
              "class_type": "UNETLoader", "_meta": {"title": "Load Diffusion"}},
        "2": {"inputs": {"model": ["1", 0], "seed": 1},
              "class_type": "KSampler", "_meta": {"title": "KSampler"}},
    }
    created = inject.inject_loras(workflow, {"A": 0.5, "B": 0.3})
    assert created == ["3", "4"]
    for nid in created:
        assert workflow[nid]["class_type"] == "LoraLoaderModelOnly"
        assert "clip" not in workflow[nid]["inputs"]
    assert workflow["4"]["inputs"]["model"] == ["3", 0]
    assert workflow["2"]["inputs"]["model"] == ["4", 0]


def test_checkpoint_chain_still_gets_a_full_lora_loader(simple_workflow):
    """The model-only detection must not regress ordinary checkpoint graphs."""
    inject.inject_loras(simple_workflow, {"MyStyleLoRA": 0.7})
    assert simple_workflow["8"]["class_type"] == "LoraLoader"
    assert simple_workflow["8"]["inputs"]["clip"] == ["1", 1]


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
# set_input_image(): skipping a fixed reference slot
# ----------------------------------------------------------------------

def _two_slot_workflow():
    """Two LoadImage nodes where node 1 is a fixed anchor, node 2 the subject.

    Node ordering is deliberately hostile: the anchor comes first, so the
    default "take the first LoadImage" behaviour picks the wrong one.
    """
    return {
        "1": {"inputs": {"image": "style_anchor_01.png"},
              "class_type": "LoadImage",
              "_meta": {"title": "Style Anchor (do not touch)"}},
        "2": {"inputs": {"image": "placeholder.png"},
              "class_type": "LoadImage",
              "_meta": {"title": "Subject"}},
    }


def test_first_load_image_is_the_anchor_without_skip_titles():
    """Baseline: the naive path writes over the reference image."""
    workflow = _two_slot_workflow()
    assert inject.set_input_image(workflow, "subject.png") == "1"
    assert workflow["1"]["inputs"]["image"] == "subject.png"  # anchor clobbered


def test_skip_titles_matches_on_node_title():
    workflow = _two_slot_workflow()
    assert inject.set_input_image(
        workflow, "subject.png", skip_titles=["style anchor"]) == "2"
    assert workflow["1"]["inputs"]["image"] == "style_anchor_01.png"
    assert workflow["2"]["inputs"]["image"] == "subject.png"


def test_skip_titles_matches_on_current_filename():
    """An untitled anchor is still recognisable from the filename it holds."""
    workflow = _two_slot_workflow()
    workflow["1"]["_meta"]["title"] = "Load Image"
    assert inject.set_input_image(
        workflow, "subject.png", skip_titles=["anchor"]) == "2"
    assert workflow["1"]["inputs"]["image"] == "style_anchor_01.png"


def test_skip_titles_ignores_a_wired_image_input():
    """A LoadImage whose `image` is a link must not blow up the haystack."""
    workflow = _two_slot_workflow()
    workflow["1"]["inputs"]["image"] = ["9", 0]
    assert inject.set_input_image(
        workflow, "subject.png", skip_titles=["style anchor"]) == "2"


def test_index_counts_surviving_candidates_only():
    workflow = _two_slot_workflow()
    workflow["3"] = {"inputs": {"image": "c.png"}, "class_type": "LoadImage",
                     "_meta": {"title": "Second Subject"}}
    # Candidates after skipping node 1 are ["2", "3"], so index 1 is node 3.
    assert inject.set_input_image(
        workflow, "x.png", index=1, skip_titles=["style anchor"]) == "3"


def test_skip_predicate():
    workflow = _two_slot_workflow()
    target = inject.set_input_image(
        workflow, "subject.png",
        skip_predicate=lambda nid, node: nid == "1")
    assert target == "2"


def test_skipping_everything_raises_rather_than_falling_back():
    workflow = _two_slot_workflow()
    with pytest.raises(NodeNotFoundError):
        inject.set_input_image(workflow, "x.png",
                               skip_titles=["anchor", "subject"])
    # Nothing was written anywhere.
    assert workflow["1"]["inputs"]["image"] == "style_anchor_01.png"
    assert workflow["2"]["inputs"]["image"] == "placeholder.png"


def test_title_hint_only_considers_surviving_candidates():
    """A skipped node cannot be resurrected by matching the title hint."""
    workflow = _two_slot_workflow()
    workflow["2"]["_meta"]["title"] = "Anchor-ish Subject"
    target = inject.set_input_image(
        workflow, "x.png", "anchor", skip_titles=["style anchor"])
    assert target == "2"


def test_customize_forwards_load_image_options():
    workflow = _two_slot_workflow()
    result = inject.customize(
        workflow,
        input_image="subject.png",
        input_image_skip_titles=["style anchor"],
    )
    assert result["1"]["inputs"]["image"] == "style_anchor_01.png"
    assert result["2"]["inputs"]["image"] == "subject.png"


# ----------------------------------------------------------------------
# batch_seeds()
# ----------------------------------------------------------------------

def test_batch_seeds_spacing_is_prime_by_default():
    seeds = inject.batch_seeds(4, 1000)
    assert seeds == [1000, 8919, 16838, 24757]
    assert inject.BATCH_SEED_SPACING == 7919


def test_batch_seeds_are_distinct_and_start_at_base():
    seeds = inject.batch_seeds(50, 12345)
    assert seeds[0] == 12345
    assert len(set(seeds)) == 50


def test_batch_seeds_random_base_when_none():
    seeds = inject.batch_seeds(3)
    assert len(set(seeds)) == 3
    assert all(0 <= s < 1_000_000_000 for s in seeds)


def test_batch_seeds_custom_spacing_and_wraparound():
    assert inject.batch_seeds(3, 0, spacing=10) == [0, 10, 20]
    # Values stay inside the modulus even when the base is near the ceiling.
    seeds = inject.batch_seeds(3, 999_999_999, spacing=10)
    assert all(0 <= s < 1_000_000_000 for s in seeds)


def test_batch_seeds_rejects_empty_batch():
    with pytest.raises(ValueError):
        inject.batch_seeds(0)


def test_batch_spacing_clears_the_within_workflow_stagger(simple_workflow):
    """The two seed axes must not collide: batch bands > sampler stagger."""
    simple_workflow["10"] = {
        "inputs": {"seed": 0, "model": ["1", 0]},
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "Refine Pass"},
    }
    used = set()
    for seed in inject.batch_seeds(5, 500):
        workflow = inject.customize(simple_workflow, seed=seed)
        for nid in nodes.find_sampler_nodes(workflow):
            value = workflow[nid]["inputs"]["seed"]
            assert value not in used
            used.add(value)


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
