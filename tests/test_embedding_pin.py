"""Guard that the encoder identity is pinned, not merely named.

The Phase 10 byte-identity tests compare digests.  A digest is only reproducible
when the encoder revision and the tokenizer stack are fixed, so this module
asserts the pin itself: the loader must pass an explicit revision, and the
recorded environment must match the running one.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from henri import semantic_backbone
from henri.semantic_backbone import MODEL_NAME, MODEL_REVISION, SemanticStem

PIN_PATH = Path(__file__).parent / "pins" / "embedding_pins.json"


def test_revision_is_pinned_to_a_commit() -> None:
    assert len(MODEL_REVISION) == 40, "a model revision must be a full commit sha"
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)


def test_model_identity_matches_the_pin_file() -> None:
    payload = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert payload["model"] == MODEL_NAME
    assert payload["revision"] == MODEL_REVISION


def test_loader_passes_the_revision_to_sentence_transformers() -> None:
    source = inspect.getsource(SemanticStem.__init__)
    assert "revision=" in source, "the loader must pass revision= explicitly"


def test_stem_info_reports_the_revision() -> None:
    field_names = {field.name for field in semantic_backbone.StemInfo.__dataclass_fields__.values()}
    assert "revision" in field_names
