# Adding a backend

The EO Inference Toolbox is built around a small, pluggable backend registry
(`eocore/backends/base.py`). Adding a new inference model is **one new module
plus one `register_backend` call** — the `load_backend` dispatch never needs
editing, and the GUI, docs and descriptors pick up the facts you register.

This guide assumes you have read [ARCHITECTURE.md](ARCHITECTURE.md); the golden
rule there applies here: *dependencies point downward, and heavy imports (torch,
your model package) happen only at call time, never at module import.*

## The contract

A backend turns one imagery tile into one embedding tile on the same pixel grid:

```
embed_tile(image[C_in, h, w], meta: TileMeta) -> embedding[D, h, w]
```

The embed algorithm windows the raster, feeds tiles here, and mosaics the
result, so your backend only has to embed a single tile and upsample its native
token/patch grid back to the tile's pixel resolution. It must satisfy the
`EmbeddingBackend` protocol: the attributes `backend_id` and `display_name`, and
the methods `descriptor()`, `required_bands()` and `embed_tile(image, meta)`.

Optionally implement `embed_tiles(images, meta)` (the `BatchEmbeddingBackend`
capability) to embed several equal-shaped tiles in one forward pass; if you do
not, `embed_tile_batch` transparently falls back to a per-tile loop, so batching
only ever changes throughput, never results.

Embeddings should **not** be unit-normalised (`is_unit_normalized = False`) — the
embed algorithm L2-normalises before writing so downstream similarity/change can
use the dot-product shortcut.

## Step 1 — write the backend module

Create `eocore/backends/mymodel.py`. Import torch and your model package
**inside methods only**, so importing the module stays cheap and torch-free:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..descriptor import EmbeddingDescriptor
from .base import backend_info, descriptor_from_info, require_model_attr, select_feature

if TYPE_CHECKING:
    from .base import TileMeta


class MyModelBackend:
    backend_id = "mymodel"
    display_name = "My Model"

    def __init__(
        self,
        *,
        weights: str | None = None,
        device: str | None = None,
        progress: Any = None,
        **_ignored: Any,
    ) -> None:
        self._weights = weights
        self._device_pref = device
        self._progress = progress  # optional weight-download callback (E-5)
        self._model: Any = None
        self._embedding_dim: int | None = None

    def descriptor(self) -> EmbeddingDescriptor:
        return descriptor_from_info(backend_info(self.backend_id), n_bands=self._embedding_dim)

    def required_bands(self) -> tuple[str, ...]:
        return backend_info(self.backend_id).default_bands

    def embed_tile(self, image: NDArray[Any], meta: TileMeta) -> NDArray[Any]:
        import torch  # lazy!

        self._ensure_model()
        # ... run the model, upsample tokens to (D, h, w), set self._embedding_dim ...
        raise NotImplementedError
```

### Guard the uncertain seams (H-6)

Any place you reach into your model's structure — an attribute path, or picking
a tensor out of its output — is where an upstream version bump can silently
break you. Use the shared helpers so failures are loud and actionable instead of
an opaque `AttributeError`:

- `require_model_attr(obj, name, backend_id, detail)` — returns `getattr(obj,
  name)` or raises a "pin a known-good version" error if it is missing/`None`.
- `select_feature(output, backend_id, index=...)` — pulls a feature tensor from a
  single-tensor-or-sequence forward-pass output, failing loud on an empty
  sequence.
- `unsupported_model_error(backend_id, detail)` — build the same error directly
  for a custom check (e.g. `if not callable(model): raise ...`).

Record the exact library + checkpoint versions you verified in
[SMOKE_TEST.md](SMOKE_TEST.md).

## Step 2 — register facts + loader

At the bottom of `base.py` (next to the built-in registrations), add a lazy
loader and one `register_backend` call. The loader is where the concrete class is
imported, so registration itself never imports torch:

```python
def _load_mymodel(**config: Any) -> EmbeddingBackend:
    from .mymodel import MyModelBackend

    return MyModelBackend(**config)


register_backend(
    BackendInfo(
        backend_id="mymodel",
        display_name="My Model",
        summary="One-line description shown on the wizard card.",
        default_bands=RGBN_BANDS,
        default_collection=S2_L2A,
        embedding_dim=512,
        resolution_m=10.0,
        is_unit_normalized=False,
        licence="Apache-2.0",
        pip_packages=("torch", "my-model-package"),
        weights_hint="Where the weights come from",
        default_input_scale=1.0 / 10000.0,
        import_names=("my_model_package",),
    ),
    _load_mymodel,
)
```

`register_backend` raises `ValueError` on a duplicate id (pass `replace=True` to
override deliberately). That is the whole wiring — `load_backend("mymodel", ...)`
now works, the wizard shows an install/status card, and the Processing "Embed"
algorithm lists it. Third-party code can call `register_backend` at import time
too; use `registered_backends()` to see everything currently registered.

## Step 3 — test it

The conformance test (`tests/test_conformance.py`) iterates every registered
backend and asserts it satisfies the `EmbeddingBackend` protocol, so a new
registration is covered automatically. Add backend-specific unit tests for any
pure-Python guard logic (see `tests/test_registry.py` for the pattern: construct
the backend, assign a fake `_model`, and assert your seam raises the actionable
error). Keep those tests torch-free so they run in the standard suite; the real
model path is verified once, by hand, in [SMOKE_TEST.md](SMOKE_TEST.md).
