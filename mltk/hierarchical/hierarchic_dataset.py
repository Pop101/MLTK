
# The job of the hierarchic dataset is to provide data in hierarchy format.
# We want to do the following
# - retrieve information about the hierarchy levels
# - define a custom sampler for all hierarchy levels
import copy
from typing import Callable, Dict, Iterator, List, NewType, Optional, Tuple, Union
from torch.utils.data import Dataset, Sampler
import torch


LevelPath = NewType("LevelPath", Tuple[Optional[int], ...])
LeafPath = NewType("LeafPath", Tuple[int, ...])


class HierarchyInformation:
    """Lightweight representation of hierarchy metadata.

    Optionally carries `leaf_centroids`, a mapping from fully-resolved leaf
    tuple -> (lat, lon) of that leaf cluster's centroid. When present, the
    hierarchical evaluator can compute Haversine distance errors against it
    without needing to thread centroid tables through every call site.
    """

    def __init__(
        self,
        hierarchy_info: Dict[Tuple, List[Tuple]],
        leaf_counts: Optional[Dict[Tuple, int]] = None,
        key_length: Optional[int] = None,
        leaf_offsets: Optional[Dict[Tuple, int]] = None,
        leaf_centroids: Optional[Dict[Tuple, Tuple[float, float]]] = None,
    ):
        if not hierarchy_info:
            raise ValueError("hierarchy_info must not be empty")

        self.hierarchy_info = hierarchy_info
        self.key_length = key_length or len(next(iter(hierarchy_info.keys())))
        self.leaf_counts = leaf_counts or {}
        self.leaf_offsets = leaf_offsets or {}
        self.leaf_centroids: Dict[Tuple, Tuple[float, float]] = leaf_centroids or {}
        self._size_cache: Dict[Tuple, int] = {}
        self._child_lookup_cache: Dict[Tuple, Dict[int, Tuple]] = {}

    def clone(self) -> "HierarchyInformation":
        """Return a deep-copied hierarchy information object."""
        return HierarchyInformation(
            copy.deepcopy(self.hierarchy_info),
            copy.deepcopy(self.leaf_counts),
            self.key_length,
            copy.deepcopy(self.leaf_offsets),
            copy.deepcopy(self.leaf_centroids),
        )

    def set_leaf_centroids(self, leaf_centroids: Dict[Tuple, Tuple[float, float]]) -> None:
        """Attach centroid coordinates for each leaf. Keys are coerced to tuples."""
        self.leaf_centroids = {tuple(k): (float(lat), float(lon)) for k, (lat, lon) in leaf_centroids.items()}

    def get_leaf_centroid(self, leaf_path: Tuple) -> Optional[Tuple[float, float]]:
        """Return (lat, lon) for a leaf path, or None if not registered."""
        return self.leaf_centroids.get(tuple(leaf_path))

    def root_level(self) -> LevelPath:
        return LevelPath(tuple([None] * self.key_length))

    def get_children(self, level: LevelPath) -> List[Tuple]:
        return self.hierarchy_info.get(tuple(level), [])

    def is_leaf(self, level: LevelPath) -> bool:
        return tuple(level) in self.leaf_counts

    def get_leaf_size(self, level: LevelPath) -> int:
        return self.leaf_counts.get(tuple(level), 0)

    def get_size_of_level(self, level: Tuple) -> int:
        """Get the total number of datapoints at and below a given hierarchy level."""
        if level not in self.hierarchy_info and level not in self.leaf_counts:
            raise ValueError(f"Level {level} not found in hierarchy information.")

        if len(level) == 0:
            raise ValueError("Level tuple cannot be empty.")
        if len(level) != self.key_length:
            raise ValueError(
                f"Level tuple length {len(level)} does not match expected length {self.key_length}."
            )

        if level in self._size_cache:
            return self._size_cache[level]

        if self.is_leaf(level):
            size = self.get_leaf_size(level)
        else:
            children = self.get_children(level)
            size = sum(self.get_size_of_level(child) for child in children)

        self._size_cache[level] = size
        return size

    def get_hierarchy_levels(self) -> List[Tuple]:
        return list(self.hierarchy_info.keys())

    def get_level_sizes(self) -> Dict[Tuple, int]:
        return copy.deepcopy(self.leaf_counts)

    def get_leaf_offset(self, level: LevelPath) -> int:
        key = tuple(level)
        if key not in self.leaf_offsets:
            raise KeyError(f"Leaf offset not found for level {key}.")
        return self.leaf_offsets[key]

    def get_leaf_index_range(self, level: LevelPath) -> Tuple[int, int]:
        offset = self.get_leaf_offset(level)
        size = self.get_leaf_size(level)
        return offset, offset + size

    def _build_child_lookup(self, level: LevelPath) -> Tuple[Dict[int, Tuple], int]:
        key = tuple(level)
        if key in self._child_lookup_cache:
            none_index = next((idx for idx, value in enumerate(level) if value is None), -1)
            return self._child_lookup_cache[key], none_index

        children = self.get_children(level)
        lookup: Dict[int, Tuple] = {}
        try:
            none_index = tuple(level).index(None)
        except ValueError:
            none_index = -1

        for child in children:
            if isinstance(child, tuple):
                # Child is a fully-specified tuple key; not addressable by an int at this depth.
                continue
            if none_index >= 0:
                new_level = list(level)
                new_level[none_index] = child
                lookup[int(child)] = tuple(new_level)

        self._child_lookup_cache[key] = lookup
        return lookup, none_index

    def advance_to_child(self, level: LevelPath, leaf_path: LeafPath) -> LevelPath:
        """Return the matching child tuple for the provided hierarchy path using cached lookups.

        Used during teacher-forced descent (training, eval loss): given the
        ground-truth leaf, pick the child that is on the path to it.
        """
        children = self.get_children(level)
        if not children:
            return level

        lookup, none_index = self._build_child_lookup(level)
        if none_index < 0:
            return level

        target_value = int(leaf_path[none_index])
        cached = lookup.get(target_value)
        if cached is not None:
            return cached
        raise ValueError(f"No child of level {tuple(level)} matches leaf path {tuple(leaf_path)}.")

    def advance_by_child(self, level: LevelPath, child) -> LevelPath:
        """Advance `level` by writing `child` into the first None slot.

        Used during free descent (beam search, greedy predict) where we have
        directly chosen a child rather than following a known leaf. ``child``
        may be either:

        - an int: the value to write into the first None slot of ``level``
        - a tuple: a fully-resolved next level (returned as-is)
        """
        if isinstance(child, tuple):
            return LevelPath(tuple(child))

        lookup, none_index = self._build_child_lookup(level)
        child_int = int(child)
        cached = lookup.get(child_int)
        if cached is not None:
            return cached

        if none_index < 0:
            raise ValueError(
                f"Cannot advance level {tuple(level)} by child {child}: no None slot to fill."
            )

        new_level = list(level)
        new_level[none_index] = child_int
        result = LevelPath(tuple(new_level))
        lookup[child_int] = result
        return result


class HierarchicDataset(HierarchyInformation, Dataset):
    """
    A dataset organized in a hierarchical manner. 
    Note that hierarchy levels must be represented as complete tuples, e.g., (level_0_id, level_1_id, ..., level_n_id).
    level 0 IS THE MOST GRANULAR LEVEL, level n IS THE MOST GENERAL/COARSE/TOP LEVEL/
    """

    def __init__(self, data: Dict[Tuple, Dataset], hierarchy_info: Dict[Tuple, List[Tuple]]):
        """Initialize the HierarchicDataset with data and hierarchy information.
        Args:
            data (Dict[Tuple, Any]): The dataset organized in a hierarchical manner. Keys: Tuple representing hierarchy levels, Values: data samples.
            hierarchy_info (Dict[Tuple, list[Tuple]]): Information about the hierarchy levels. Keys: Tuple representing hierarchy levels, Values: list of tuples indicating all children of this level.
        
        Note: This method does no validation. We expect:
        - data keys are COMPLETE tuples (none have null/missing levels)
        - hierarchy_info keys are nullable and in right-triangle form (e.g., (None, None), (None, 1), (1, 2), etc.)
        """
        key_length = len(next(iter(data.keys())))

        def _to_leaf_key(raw_key: Tuple) -> LeafPath:
            leaf_key = LeafPath(tuple(int(v) for v in raw_key))
            if len(leaf_key) != key_length:
                raise ValueError(f"LeafPath length {len(leaf_key)} != expected {key_length}: {leaf_key}")
            return leaf_key

        def _to_level_key(raw_level: Tuple) -> LevelPath:
            level = LevelPath(tuple(None if v in (None, "") else int(v) for v in raw_level))
            if len(level) != key_length:
                raise ValueError(f"LevelPath length {len(level)} != expected {key_length}: {level}")
            return level

        def _normalize_children(raw_children: List[int]) -> List[Tuple]:
            children: List[Tuple] = []
            for child in raw_children:
                if isinstance(child, tuple):
                    children.append(tuple(_to_level_key(child)))
                else:
                    children.append(int(child))
            return children

        # Normalize keys into strong types so downstream users don't need coercion.
        self.data = {_to_leaf_key(k): v for k, v in data.items()}

        # Normalize hierarchy_info into a LevelPath -> children mapping.
        normalized_hierarchy_info: Dict[LevelPath, List[Tuple]] = {
            _to_level_key(level): _normalize_children(children)
            for level, children in hierarchy_info.items()
        }
        leaf_counts = {tuple(level): len(samples) for level, samples in self.data.items()}
        self._cumulative_counts = {}
        self._count = 0
        for hpath in self.data.keys():
            self._cumulative_counts[tuple(hpath)] = self._count
            self._count += len(self.data[hpath])

        super().__init__(
            {tuple(k): v for k, v in normalized_hierarchy_info.items()},
            leaf_counts=leaf_counts,
            key_length=key_length,
            leaf_offsets=self._cumulative_counts,
        )


    def __len__(self):
        """Get the total number of datapoints in the dataset."""
        return self._count
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Tuple[float, float], LeafPath, int]:
        """Retrieve a datapoint by its global index."""
        if idx < 0 or idx >= self._count:
            raise IndexError("Index out of range.")
        
        # Find the correct hierarchy path for the given index
        for hpath, start_idx in self._cumulative_counts.items():
            end_idx = start_idx + len(self.data[hpath])
            if start_idx <= idx < end_idx:
                local_idx = idx - start_idx
                sample = self.data[hpath][local_idx]
                if isinstance(sample, tuple):
                    # Expected sample format: (feature_tensor, output_val)
                    tensor, output_val = sample
                    return tensor, output_val, self.get_leaf_path(idx), idx
                raise TypeError(
                    "Leaf datasets must return (tensor, (lat, lon)) tuples. "
                    f"Got type={type(sample)} at idx={idx}."
                )
        
        raise IndexError("Index not found in dataset.")
    
    def get_leaf_path(self, idx: int) -> LeafPath:
        """Get the LeafPath for a given global index."""
        if idx < 0 or idx >= self._count:
            raise IndexError("Index out of range.")
        
        for hpath, start_idx in self._cumulative_counts.items():
            end_idx = start_idx + len(self.data[hpath])
            if start_idx <= idx < end_idx:
                return LeafPath(tuple(int(v) for v in hpath))
        
        raise IndexError("Index not found in dataset.")
    
    def get_indices_for_level(self, level: Tuple) -> List[int]:
        """Get all global indices that belong to a specific hierarchy level."""
        if level not in self.data:
            return []
        
        start_idx = self._cumulative_counts[level]
        end_idx = start_idx + len(self.data[level])
        return list(range(start_idx, end_idx))
    
    def get_hierarchy_levels(self) -> List[Tuple]:
        """Get all hierarchy levels (keys) in the dataset."""
        return list(self.data.keys())
    
    def get_level_sizes(self) -> Dict[Tuple, int]:
        """Get the size of each hierarchy level."""
        return self.leaf_counts.copy()

    def to_hierarchy_information(self) -> HierarchyInformation:
        """Return a lightweight hierarchy information copy without dataset references."""
        return self.clone()
    
class PerLevelSampler(Sampler):
    """A sampler that adapts other torch samplers to work with hierarchical datasets.

    For each hierarchy level, it creates a separate sampler using the provided
    factory function. Each time a sample is requested, we traverse down the
    hierarchy, sampling at each level to get to the next.
    """

    def __init__(
        self,
        dataset: HierarchicDataset,
        sampler_factory: Callable[[List[int]], Sampler],
        num_samples: Optional[int] = None,
        cycle_on_exhaustion: bool = True,
    ):
        """
        Args:
            dataset: The HierarchicDataset to sample from. Needed to know hierarchy structure.
            sampler_factory: Factory function to create a sampler for each level.
                Called with a list of labels (one per item in the level of the
                hierarchy), and should return a Sampler that can sample among
                those labels.
            num_samples: Total number of samples to generate per epoch.
            cycle_on_exhaustion: when True (default), restart per-level
                iterators on ``StopIteration``; when False, end the epoch the
                first time any per-level sampler is exhausted.
        """
        self.dataset = dataset
        self.sampler_factory = sampler_factory
        self.num_samples = num_samples or len(dataset)
        self.cycle_on_exhaustion = cycle_on_exhaustion

        self._level_samplers: Dict[Tuple, Sampler] = {}
        self._sampler_iters: Dict[Tuple, Iterator] = {}
    
    def _create_sampler_for_level(self, level_path: Tuple) -> Sampler:
        """Lazily create a sampler for a given level."""
        # Check if this is a leaf level (actual data indices)
        if level_path in self.dataset.data:
            indices = self.dataset.get_indices_for_level(level_path)
            return self.sampler_factory(indices)
        
        # Otherwise it's a hierarchy level (children to sample from)
        children = self.dataset.hierarchy_info.get(level_path, [])
        if not children:
            raise ValueError(f"No children found for hierarchy level {level_path}.")
        return self.sampler_factory(children)
    
    def _get_or_make_sampler(self, level_path: Tuple) -> Sampler:
        if level_path not in self._level_samplers:
            self._level_samplers[level_path] = self._create_sampler_for_level(level_path)
        return self._level_samplers[level_path]

    def _get_sampler_iter(self, level_path: Tuple) -> Iterator:
        if level_path not in self._sampler_iters:
            self._sampler_iters[level_path] = iter(self._get_or_make_sampler(level_path))
        return self._sampler_iters[level_path]

    def _next_or_cycle(self, level_path: Tuple) -> Optional[int]:
        """Pull the next index from ``level_path``'s sampler. On exhaustion,
        cycle (default) or return ``None`` to end the epoch."""
        try:
            return next(self._get_sampler_iter(level_path))
        except StopIteration:
            if not self.cycle_on_exhaustion:
                return None
            self._sampler_iters[level_path] = iter(self._get_or_make_sampler(level_path))
            try:
                return next(self._sampler_iters[level_path])
            except StopIteration:
                return None

    def _sample_hierarchy_path(self, current_level: Tuple) -> Optional[Tuple]:
        """Sample a complete hierarchy path from ``current_level`` to a leaf."""
        for _ in range(self.dataset.key_length):
            none_index = current_level.index(None)
            children = self.dataset.hierarchy_info[current_level]
            child_idx = self._next_or_cycle(current_level)
            if child_idx is None:
                return None
            new_level = list(current_level)
            new_level[none_index] = children[child_idx]
            current_level = tuple(new_level)
        return current_level

    def __iter__(self):
        top_level = tuple([None] * self.dataset.key_length)
        for _ in range(self.num_samples):
            leaf_path = self._sample_hierarchy_path(top_level)
            if leaf_path is None:
                break
            idx = self._next_or_cycle(leaf_path)
            if idx is None:
                break
            yield idx
    
    def __len__(self):
        """Return the number of samples per epoch."""
        return self.num_samples
