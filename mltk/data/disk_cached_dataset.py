"""On-disk, mmap-backed, chunked, randomly-indexed dataset cache.

Use case: precompute frozen-backbone features (or anything else slow) once
and serve from disk for every subsequent epoch.

Design goals
------------
1. **mmap.** Reads bypass Python copies; the OS page cache decides what
   stays in RAM. Files are laid out as raw little-endian tensor data with
   a JSON manifest describing dtype + shape; ``numpy.memmap`` reads them
   directly.
2. **Chunked.** Items are grouped into ``chunk_size``-sized files. On HDDs
   this turns scattered random reads into long sequential reads when the
   access pattern is in-order; on SSDs the cost is essentially zero. Each
   chunk is a separate file, mmap'd independently and on demand.
3. **O(1) random access.** Item ``i`` lives at chunk ``i // chunk_size``,
   slot ``i % chunk_size``. No index file scan, no hash lookup.
4. **Resilient to interrupted fills.** A bitmap tracks which items have
   been computed. The first epoch may be partial; on restart, missing
   items are re-computed from the source dataset on demand. The bitmap
   itself is mmap'd, so flushes are basically free.
5. **Schema-locked on first item.** First successful fetch from the source
   determines the (image dtype, image shape, label dtype, label shape) for
   the whole cache — written into the manifest. Subsequent items must
   match; mismatch raises.

Storage layout
--------------
::

    <cache_dir>/
        manifest.json        - schema + chunk geometry
        filled.bin           - uint8 bitmap, n_items bytes (1 = filled)
        chunks/
            00000.image.bin  - chunk 0 image tensors, mmap'd
            00000.label.bin  - chunk 0 label tensors, mmap'd
            00001.image.bin  - chunk 1, etc.
            ...

Constraints
-----------
- Items must be ``(image_tensor, label_tensor)``. Both must be torch
  tensors. Both must have a fixed dtype and shape across the whole cache.
- Single-process writers. Cross-process safety would need flock; punted.

Composition with ``MapDataset``
-------------------------------
::

    feats   = MapDataset(image_ds, fn=lambda img, lbl: (backbone(img), lbl))
    cached  = DiskCachedDataset(feats, cache_dir="cache/dino_features")
    loader  = DataLoader(cached, batch_size=64, shuffle=True)

The first epoch fills the cache (slow); every subsequent epoch reads
straight from mmap (fast).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# Map between numpy dtype strings and torch dtypes. Limited on purpose to
# the subset that round-trips cleanly through both libraries' memmap path.
_NP_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_TORCH_TO_NP = {v: k for k, v in _NP_TO_TORCH.items()}

_MANIFEST_VERSION = 1


def _torch_dtype_to_str(dt: torch.dtype) -> str:
    if dt not in _TORCH_TO_NP:
        raise TypeError(
            f"DiskCachedDataset can't serialize torch dtype {dt!r}; "
            f"supported: {sorted(_TORCH_TO_NP.keys(), key=lambda d: str(d))}"
        )
    return _TORCH_TO_NP[dt]


def _str_to_np_dtype(name: str) -> np.dtype:
    if name not in _NP_TO_TORCH:
        raise ValueError(f"unsupported dtype name in manifest: {name!r}")
    return np.dtype(name)


class DiskCachedDataset(Dataset):
    """Caches a Dataset's items on disk with mmap, chunking, and O(1) random
    access. See module docstring for layout and rationale.

    Args:
        dataset: source Dataset producing ``(image_tensor, label_tensor)``
                 items. The cache fills lazily on first access of each idx.
        cache_dir: directory to store manifest + chunk files. Created if
                   missing. Reused on next run if the manifest matches.
        chunk_size: items per chunk file. Larger chunks = better HDD
                    throughput; smaller chunks = lower mmap overhead.
                    Default 512 is a reasonable middle.
        verify_schema: when reusing an existing cache, refetch one source
                       item and assert its shape/dtype matches the manifest.
                       Off by default — saves one source fetch on warm
                       starts. Turn on if you suspect drift.
    """

    MANIFEST_NAME = "manifest.json"
    BITMAP_NAME = "filled.bin"
    CHUNKS_DIR = "chunks"

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Union[str, os.PathLike],
        chunk_size: int = 512,
        verify_schema: bool = False,
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / self.CHUNKS_DIR).mkdir(parents=True, exist_ok=True)

        # Source-driven length. Locked in the manifest; if the manifest
        # disagrees, manifest wins for layout but len() reflects manifest.
        self._n_items: int = len(dataset)
        self._chunk_size: int = int(chunk_size)
        self._manifest: Optional[Dict] = None
        self._bitmap: Optional[np.memmap] = None
        # chunk_id -> (image_memmap, label_memmap). Opened lazily, kept open.
        self._chunk_mmaps: Dict[int, Tuple[np.memmap, np.memmap]] = {}

        manifest_path = self.cache_dir / self.MANIFEST_NAME
        if manifest_path.exists():
            self._load_manifest()
            if verify_schema:
                self._verify_schema_against_source()
        # else: schema is unknown until the first successful source fetch.

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._n_items

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < self._n_items):
            raise IndexError(f"index {idx} out of range for n_items={self._n_items}")
        if self._is_filled(idx):
            return self._read(idx)
        # Cache miss. Compute, store, return.
        item = self.dataset[idx]
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(
                "DiskCachedDataset expects source to return (image, label) tuples; "
                f"got {type(item).__name__} at idx={idx}"
            )
        image, label = item
        if not torch.is_tensor(image) or not torch.is_tensor(label):
            raise TypeError(
                "DiskCachedDataset expects (tensor, tensor) items; "
                f"got ({type(image).__name__}, {type(label).__name__}) at idx={idx}"
            )
        if self._manifest is None:
            self._init_manifest(image, label)
        self._write(idx, image, label)
        return image, label

    def is_fully_filled(self) -> bool:
        """True iff every item has been computed at least once. Useful as a
        gate for ``verify_schema=True`` callers."""
        if self._manifest is None or self._bitmap is None:
            return False
        return bool(self._bitmap.all())

    def fill_progress(self) -> Tuple[int, int]:
        """``(filled, total)`` count. Cheap; reads the mmap'd bitmap."""
        if self._bitmap is None:
            return 0, self._n_items
        return int(self._bitmap.sum()), self._n_items

    def precompute(self, indices: Optional[List[int]] = None, verbose: bool = False) -> None:
        """Fill missing entries eagerly. Useful as a precompute step before
        the first training epoch so the I/O cost is paid once up-front
        instead of mid-batch.

        Args:
            indices: subset to ensure-filled. Defaults to all items.
            verbose: print a one-line progress summary every 5%.
        """
        if indices is None:
            indices = list(range(self._n_items))
        total = len(indices)
        next_log = max(1, total // 20) if verbose else 0
        for i, idx in enumerate(indices, 1):
            self.__getitem__(idx)
            if verbose and (i == total or (next_log and i % next_log == 0)):
                done, n = self.fill_progress()
                print(f"[DiskCachedDataset] precompute {i}/{total} | overall {done}/{n}")

    # ------------------------------------------------------------------ #
    # Manifest + bitmap
    # ------------------------------------------------------------------ #
    def _manifest_path(self) -> Path:
        return self.cache_dir / self.MANIFEST_NAME

    def _bitmap_path(self) -> Path:
        return self.cache_dir / self.BITMAP_NAME

    def _chunk_paths(self, chunk_id: int) -> Tuple[Path, Path]:
        chunks = self.cache_dir / self.CHUNKS_DIR
        return (
            chunks / f"{chunk_id:05d}.image.bin",
            chunks / f"{chunk_id:05d}.label.bin",
        )

    def _load_manifest(self) -> None:
        with open(self._manifest_path(), "r") as f:
            mf = json.load(f)
        if mf.get("version") != _MANIFEST_VERSION:
            raise RuntimeError(
                f"DiskCachedDataset manifest version {mf.get('version')} "
                f"!= expected {_MANIFEST_VERSION}; clear {self.cache_dir} to rebuild"
            )
        if mf["n_items"] != self._n_items:
            raise RuntimeError(
                f"DiskCachedDataset n_items mismatch: source has {self._n_items}, "
                f"cache has {mf['n_items']}; clear {self.cache_dir} to rebuild"
            )
        self._manifest = mf
        self._chunk_size = mf["chunk_size"]
        # Open or create the fill bitmap in r+ mode.
        bm_path = self._bitmap_path()
        if not bm_path.exists():
            # Defensive: manifest exists but bitmap missing => treat as empty.
            with open(bm_path, "wb") as f:
                f.write(b"\x00" * self._n_items)
        self._bitmap = np.memmap(bm_path, dtype=np.uint8, mode="r+", shape=(self._n_items,))

    def _init_manifest(self, image: torch.Tensor, label: torch.Tensor) -> None:
        n_chunks = (self._n_items + self._chunk_size - 1) // self._chunk_size
        manifest = {
            "version": _MANIFEST_VERSION,
            "n_items": self._n_items,
            "chunk_size": self._chunk_size,
            "n_chunks": n_chunks,
            "image": {
                "dtype": _torch_dtype_to_str(image.dtype),
                "shape": list(image.shape),
            },
            "label": {
                "dtype": _torch_dtype_to_str(label.dtype),
                "shape": list(label.shape),
            },
        }
        # Atomic manifest write (tmp + rename) so a crash mid-init can't
        # leave a half-written manifest.
        mp = self._manifest_path()
        tmp = mp.with_suffix(mp.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, mp)
        # Bitmap of zeros, length n_items.
        bm_path = self._bitmap_path()
        with open(bm_path, "wb") as f:
            f.write(b"\x00" * self._n_items)
        self._bitmap = np.memmap(bm_path, dtype=np.uint8, mode="r+", shape=(self._n_items,))
        self._manifest = manifest

    def _verify_schema_against_source(self) -> None:
        if self._n_items == 0:
            return
        sample = self.dataset[0]
        if not (isinstance(sample, tuple) and len(sample) == 2):
            raise TypeError("verify_schema: source did not return a (image, label) tuple")
        image, label = sample
        mf = self._manifest
        if mf is None:
            raise RuntimeError("verify_schema: manifest is not initialized")
        if list(image.shape) != mf["image"]["shape"]:
            raise RuntimeError(
                f"verify_schema: image shape {list(image.shape)} != "
                f"manifest {mf['image']['shape']}"
            )
        if _torch_dtype_to_str(image.dtype) != mf["image"]["dtype"]:
            raise RuntimeError(
                f"verify_schema: image dtype {image.dtype} != manifest {mf['image']['dtype']}"
            )
        if list(label.shape) != mf["label"]["shape"]:
            raise RuntimeError(
                f"verify_schema: label shape {list(label.shape)} != "
                f"manifest {mf['label']['shape']}"
            )
        if _torch_dtype_to_str(label.dtype) != mf["label"]["dtype"]:
            raise RuntimeError(
                f"verify_schema: label dtype {label.dtype} != manifest {mf['label']['dtype']}"
            )

    # ------------------------------------------------------------------ #
    # Chunk mmaps — opened lazily, kept open for the dataset lifetime
    # ------------------------------------------------------------------ #
    def _chunk_item_count(self, chunk_id: int) -> int:
        """Items in this chunk. The last chunk is short when n_items isn't
        a multiple of chunk_size; allocating that file at full chunk_size
        wastes disk for nothing."""
        start = chunk_id * self._chunk_size
        return min(self._chunk_size, self._n_items - start)

    def _open_chunk(self, chunk_id: int) -> Tuple[np.memmap, np.memmap]:
        cached = self._chunk_mmaps.get(chunk_id)
        if cached is not None:
            return cached
        mf = self._manifest
        if mf is None:
            raise RuntimeError("manifest must exist before opening chunks")

        img_path, lbl_path = self._chunk_paths(chunk_id)
        img_dtype = _str_to_np_dtype(mf["image"]["dtype"])
        lbl_dtype = _str_to_np_dtype(mf["label"]["dtype"])
        img_shape = (self._chunk_item_count(chunk_id), *mf["image"]["shape"])
        lbl_shape = (self._chunk_item_count(chunk_id), *mf["label"]["shape"])

        # Pre-size the file if it doesn't exist. ``numpy.memmap(... mode='w+')``
        # creates and zero-fills; ``mode='r+'`` opens existing.
        img_mode = "r+" if img_path.exists() else "w+"
        lbl_mode = "r+" if lbl_path.exists() else "w+"
        img_mm = np.memmap(img_path, dtype=img_dtype, mode=img_mode, shape=img_shape)
        lbl_mm = np.memmap(lbl_path, dtype=lbl_dtype, mode=lbl_mode, shape=lbl_shape)
        self._chunk_mmaps[chunk_id] = (img_mm, lbl_mm)
        return img_mm, lbl_mm

    # ------------------------------------------------------------------ #
    # Read / write paths
    # ------------------------------------------------------------------ #
    def _is_filled(self, idx: int) -> bool:
        if self._bitmap is None:
            return False
        return bool(self._bitmap[idx])

    def _read(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_id, slot = divmod(idx, self._chunk_size)
        img_mm, lbl_mm = self._open_chunk(chunk_id)
        # ``.copy()`` detaches from the mmap-backed buffer, so subsequent
        # tensor ops can't surprise the OS by mutating page-cache pages.
        # Cost: one memcpy per access; tiny next to forward()/backward().
        img_np = np.array(img_mm[slot], copy=True)
        lbl_np = np.array(lbl_mm[slot], copy=True)
        return torch.from_numpy(img_np), torch.from_numpy(lbl_np)

    def _write(self, idx: int, image: torch.Tensor, label: torch.Tensor) -> None:
        mf = self._manifest
        if mf is None:
            raise RuntimeError("manifest must exist before writing chunks")
        # Schema enforcement on every write — silent shape drift would
        # corrupt the cache silently.
        if list(image.shape) != mf["image"]["shape"]:
            raise RuntimeError(
                f"item {idx} image shape {list(image.shape)} != "
                f"locked manifest shape {mf['image']['shape']}"
            )
        if _torch_dtype_to_str(image.dtype) != mf["image"]["dtype"]:
            raise RuntimeError(
                f"item {idx} image dtype {image.dtype} != "
                f"locked manifest dtype {mf['image']['dtype']}"
            )
        if list(label.shape) != mf["label"]["shape"]:
            raise RuntimeError(
                f"item {idx} label shape {list(label.shape)} != "
                f"locked manifest shape {mf['label']['shape']}"
            )
        if _torch_dtype_to_str(label.dtype) != mf["label"]["dtype"]:
            raise RuntimeError(
                f"item {idx} label dtype {label.dtype} != "
                f"locked manifest dtype {mf['label']['dtype']}"
            )

        chunk_id, slot = divmod(idx, self._chunk_size)
        img_mm, lbl_mm = self._open_chunk(chunk_id)
        # ``contiguous().cpu().numpy()`` is the safe assignable view; on
        # GPU tensors we have to land on CPU before the mmap copy anyway.
        img_mm[slot] = image.detach().contiguous().cpu().numpy()
        lbl_mm[slot] = label.detach().contiguous().cpu().numpy()
        # Mark filled. Single-byte write, naturally atomic on the OS side.
        if self._bitmap is None:
            raise RuntimeError("bitmap must exist before marking cache entries filled")
        self._bitmap[idx] = 1
        # Flushing on every write is overkill on SSDs and slow on HDDs;
        # numpy.memmap will write back on close + on OS pressure. Callers
        # who want durability after a specific batch can call ``.flush()``.

    def flush(self) -> None:
        """Force every open chunk + the bitmap to disk. Call before
        relying on the cache surviving a crash."""
        if self._bitmap is not None:
            self._bitmap.flush()
        for img_mm, lbl_mm in self._chunk_mmaps.values():
            img_mm.flush()
            lbl_mm.flush()

    def close(self) -> None:
        """Release every mmap. After ``close`` the dataset is unusable."""
        self.flush()
        # numpy.memmap doesn't expose ``close``; dropping references and
        # letting the GC reap them is the canonical way.
        self._chunk_mmaps.clear()
        self._bitmap = None
