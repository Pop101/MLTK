"""HierarchyInformation + PerLevelSampler on a hand-built 2-level tree."""
import torch
from torch.utils.data import RandomSampler

from mltk import HierarchyInformation, HierarchicDataset, PerLevelSampler, create_sqrt_sampler


def _tiny_tree():
    """A 2-level tree: root -> 3 branches, each with 2 leaves of 2 samples."""
    # Use torch tensor pairs as "samples" so dataset returns non-empty tuples.
    data = {
        (0, 0): [(torch.zeros(4), (0.0, 0.0))] * 2,
        (0, 1): [(torch.zeros(4), (0.0, 1.0))] * 2,
        (1, 2): [(torch.zeros(4), (1.0, 2.0))] * 2,
        (1, 3): [(torch.zeros(4), (1.0, 3.0))] * 2,
        (2, 4): [(torch.zeros(4), (2.0, 4.0))] * 2,
        (2, 5): [(torch.zeros(4), (2.0, 5.0))] * 2,
    }
    hierarchy_info = {
        (None, None): [0, 1, 2],  # root has 3 c0 children
        (0, None): [0, 1],
        (1, None): [2, 3],
        (2, None): [4, 5],
    }
    return data, hierarchy_info


def test_hierarchy_info_basic_queries():
    _, hinfo = _tiny_tree()
    info = HierarchyInformation(hinfo, leaf_counts={(0, 0): 2, (0, 1): 2, (1, 2): 2, (1, 3): 2, (2, 4): 2, (2, 5): 2})
    assert info.key_length == 2
    assert info.is_leaf((0, 0))
    assert not info.is_leaf((0, None))
    assert info.get_leaf_size((0, 0)) == 2
    assert set(info.get_children((None, None))) == {0, 1, 2}


def test_hierarchic_dataset_yields_right_shape():
    data, hinfo = _tiny_tree()
    ds = HierarchicDataset(data, hinfo)
    assert len(ds) == 12
    item = ds[0]
    # (tensor, output_val, LeafPath, idx) per HierarchicDataset contract.
    assert len(item) == 4
    assert isinstance(item[-1], int)


def test_advance_to_child_reaches_leaf():
    data, hinfo = _tiny_tree()
    ds = HierarchicDataset(data, hinfo)
    info = ds.to_hierarchy_information()
    # Start from root, teacher-force toward leaf (1, 3).
    lvl = info.root_level()
    for _ in range(info.key_length):
        lvl = info.advance_to_child(lvl, (1, 3))
    assert tuple(lvl) == (1, 3)


def test_per_level_sampler_yields_valid_indices():
    data, hinfo = _tiny_tree()
    ds = HierarchicDataset(data, hinfo)
    sampler = PerLevelSampler(ds, sampler_factory=create_sqrt_sampler, num_samples=30)
    idxs = list(sampler)
    assert len(idxs) == 30
    for i in idxs:
        assert 0 <= i < len(ds)
