"""Shared fixtures. No ComfyUI server is needed for any test in this suite."""

import json
import sys
from pathlib import Path

import pytest

# Allow running the tests straight from a checkout, without installing first.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a workflow fixture by filename."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def simple_workflow() -> dict:
    """A minimal SD txt2img workflow in API format, fresh for each test."""
    return load_fixture("simple_workflow.json")


@pytest.fixture
def untitled_workflow() -> dict:
    """The same workflow with ComfyUI's ambiguous default encoder titles.

    Both text encoders are called "CLIP Text Encode (Prompt)", which is what
    you get when nobody renames anything. Only the graph edges distinguish the
    positive encoder from the negative one.
    """
    workflow = load_fixture("simple_workflow.json")
    for node_id in ("2", "3"):
        workflow[node_id]["_meta"]["title"] = "CLIP Text Encode (Prompt)"
    return workflow
