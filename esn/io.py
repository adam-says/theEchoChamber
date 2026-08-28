"""Public runtime artifact loading.

Legacy pickle handling intentionally lives only in ``tools/migrate_legacy_model.py``.
Importing the live runtime therefore cannot deserialize executable objects.
"""

from __future__ import annotations

from .artifact import load_inference_artifact
from .streaming import InferenceStreamer


def load_artifact(path: str, *, backend: str = "auto") -> InferenceStreamer:
    """Load a versioned numeric artifact and construct the streaming runtime."""
    if str(path).lower().endswith(".npz"):
        return InferenceStreamer(load_inference_artifact(path), backend=backend)  # type: ignore[arg-type]
    raise ValueError(
        "pickle ESN artifacts are no longer accepted by the runtime; "
        "migrate or retrain the model to the versioned .npz format"
    )

