import os

import numpy as np
import numpy.typing as npt

class Index:
    """Mutable HNSW index over float32 vectors, squared L2 distance."""

    def __init__(
        self, dim: int, M: int = 16, ef_construction: int = 200, seed: int = 100
    ) -> None: ...
    def add(self, vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        """Add vectors (float32, C-contiguous, shape (n, dim) or (dim,)); return their ids."""
    def search(
        self,
        queries: npt.NDArray[np.float32],
        k: int,
        ef: int,
        threads: int = 1,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """k nearest neighbors per query: (ids, SQUARED L2 distances); missing results are -1 and +inf."""
    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the index to path atomically."""
    @property
    def size(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def M(self) -> int: ...
    @property
    def ef_construction(self) -> int: ...
    @property
    def seed(self) -> int: ...

class LoadedIndex:
    """Read-only index loaded by load()."""

    def search(
        self,
        queries: npt.NDArray[np.float32],
        k: int,
        ef: int,
        threads: int = 1,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """k nearest neighbors per query: (ids, SQUARED L2 distances); missing results are -1 and +inf."""
    def close(self) -> None:
        """Release the mapping or buffer. Idempotent."""
    @property
    def closed(self) -> bool: ...
    @property
    def size(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def __enter__(self) -> LoadedIndex: ...
    def __exit__(self, *args: object) -> None: ...

def load(
    path: str | os.PathLike[str],
    mode: str = "mmap",
    validate: str = "deep",
) -> LoadedIndex:
    """Load a saved index. mode: "mmap" or "read"; validate: "deep" or "shallow" (trusted files only)."""
