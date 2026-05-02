"""Prompt templates used to elicit concept-specific vs. background activations.

Templates follow the paper's Table 6 / 7. For each target concept *c*, FIA pairs:

* a **concept prompt** that embeds *c* in a neutral context, and
* a **base prompt** that describes only the surrounding context.

The contrastive saliency in :mod:`fia.saliency` then isolates neurons that fire for
*c* itself rather than for the background.
"""

from __future__ import annotations

import os
import random
from typing import Iterable

# ---------------------------------------------------------------------------
# Placeholder vocabularies (Table 7)
# ---------------------------------------------------------------------------

PLACES = [
    "road", "tree", "forest", "lawn", "clubhouse", "courtyard", "backyard",
    "cityscape", "suburb", "mall", "cafe", "office", "library", "market",
    "bridge", "harbor", "garden", "beach", "room", "park", "street",
    "shelter", "chair", "table", "bag", "mountain", "valley", "waterfall",
    "desert", "sunrise",
]

PERSONS = [
    "man", "woman", "girl", "boy", "mother", "father", "kid", "professor",
    "student", "group of friends", "celebrity", "child", "couple", "guy",
    "doctor", "nurse", "teacher", "lawyer",
]

OBJECTS = [
    "cat", "dog", "mouse", "bear", "car", "chair", "bag", "building",
    "chicken", "duck", "sofa", "table", "tree", "bicycle", "door", "rabbit",
    "ball", "bat", "horse", "bird", "flower", "bowl", "bottle", "wall",
    "clock", "television", "guitar", "truck", "laptop", "book",
]


# ---------------------------------------------------------------------------
# Per-task prompt builders (Table 6)
# ---------------------------------------------------------------------------


def _alternating(template_a: str, template_b: str, vocab: Iterable[str], concept: str):
    out = []
    for i, item in enumerate(vocab):
        tmpl = template_a if i % 2 == 0 else template_b
        out.append(tmpl.format(concept=concept, slot=item))
    return out


def object_prompts(concept: str, n: int = 30) -> tuple[list[str], list[str]]:
    """Object task: paper Table 6.

    Concept: ``"a {concept} in a/the {place}"`` / ``"a {concept} near the {place}"``
    Base:    ``"a {place}"``
    """

    places = PLACES[:n]
    concept_prompts = _alternating(
        "a {concept} in a {slot}", "a {concept} near the {slot}", places, concept
    )
    base_prompts = [f"a {p}" for p in places]
    return concept_prompts, base_prompts


def explicit_prompts(concept: str, n: int = 18) -> tuple[list[str], list[str]]:
    """Explicit-content task. ``concept`` is typically ``"naked"`` or ``"sexual"``.

    Concept: ``"a photo of a {concept} {person}"``
    Base:    ``"a photo of a {person}"``
    """

    persons = PERSONS[:n]
    concept_prompts = [f"a photo of a {concept} {p}" for p in persons]
    base_prompts = [f"a photo of a {p}" for p in persons]
    return concept_prompts, base_prompts


def art_prompts(artist: str, n: int = 30) -> tuple[list[str], list[str]]:
    """Artist-style task. ``artist`` is a painter's name.

    Concept: ``"a {object} in the style of {artist}"``
    Base:    ``"a {object}"``
    """

    objects = OBJECTS[:n]
    concept_prompts = [f"a {o} in the style of {artist}" for o in objects]
    base_prompts = [f"a {o}" for o in objects]
    return concept_prompts, base_prompts


def build_pair_prompts(task: str, concept: str, n: int = 30,
                       seed: int = 0) -> tuple[list[str], list[str]]:
    """Dispatch helper: returns ``(concept_prompts, base_prompts)`` for the task."""

    rng = random.Random(seed)
    if task == "object":
        c, b = object_prompts(concept, n=n)
    elif task == "explicit":
        c, b = explicit_prompts(concept, n=n)
    elif task == "art":
        c, b = art_prompts(concept, n=n)
    else:
        raise ValueError(f"Unknown task '{task}'")

    # Stable shuffle so the concept/base order matches across calls.
    paired = list(zip(c, b))
    rng.shuffle(paired)
    c, b = zip(*paired)
    return list(c), list(b)


# ---------------------------------------------------------------------------
# Optional file-based vocabularies (compat with the paper's exact prompt set)
# ---------------------------------------------------------------------------


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def vocab_from_file(name: str, datasets_dir: str = "datasets") -> list[str]:
    """Load a vocabulary file (``things.txt``, ``humans.txt``, ``common_scenes.txt``)."""

    return _read_lines(os.path.join(datasets_dir, name))
