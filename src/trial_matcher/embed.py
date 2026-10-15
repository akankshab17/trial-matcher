"""Turn trials into the short texts we embed, and wrap the local embedding model."""

import hashlib
from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LENGTH = 256  # tokens; longer text is cut (trial texts average ~70, p95 ~120)

# Which text describes a trial.
BASIC = "basic"  # title + conditions + interventions
WITH_INCLUSION = "with_inclusion"  # basic + the first few inclusion criteria (often stage/biomarkers)
VARIANTS = (BASIC, WITH_INCLUSION)
INCLUSION_CRITERIA_USED = 5


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """One unit-length vector per text."""


def build_trial_text(
    variant: str,
    title: str,
    conditions: Sequence[str],
    interventions: Sequence[str],
    inclusion_criteria: Sequence[str] = (),
) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    parts = [title.strip().rstrip(".")]
    if conditions:
        parts.append("Conditions: " + "; ".join(conditions))
    if interventions:
        parts.append("Interventions: " + "; ".join(interventions))
    if variant == WITH_INCLUSION and inclusion_criteria:
        # nested criteria are stored as "parent\n- child"; flatten to one line each
        flat = [" ".join(c.split()) for c in inclusion_criteria[:INCLUSION_CRITERIA_USED]]
        parts.append("Inclusion: " + " ".join(flat))
    return ". ".join(parts)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector text form, e.g. '[0.1,0.2]' (use with a ::vector cast)."""
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


class SentenceTransformerEncoder:
    """A local sentence-transformers model (uses the Apple GPU when available)."""

    def __init__(self, model_name: str):
        # Imported here so that importing this module (and running most tests) never loads torch.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = MAX_SEQ_LENGTH

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()
