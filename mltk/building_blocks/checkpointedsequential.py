from typing import Iterable, Optional, Union

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)


class CheckpointedSequential(nn.Module):
    """A Sequential-like module that applies gradient checkpointing to specified layers."""

    def __init__(
        self,
        *layers: nn.Module,
        checkpoint_segments: Optional[Union[int, Iterable[int]]] = None,
    ):
        super().__init__()

        if checkpoint_segments is None:
            checkpoint_segments = [
                i for i, layer in enumerate(layers)
                if sum(p.numel() for p in layer.parameters()) > 1_000_000
            ]
        elif isinstance(checkpoint_segments, int):
            checkpoint_segments = [checkpoint_segments]
        else:
            checkpoint_segments = list(checkpoint_segments)

        self.checkpoint_segments = checkpoint_segments
        
        # Use numbered attributes like nn.Sequential for state_dict compatibility
        for i, layer in enumerate(layers):
            if i in self.checkpoint_segments:
                checkpointed_layer = checkpoint_wrapper(
                    layer,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
                self.add_module(str(i), checkpointed_layer)
            else:
                self.add_module(str(i), layer)
    
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x
    
    def get_checkpointed_indices(self):
        """Helper method to see which layer indices are checkpointed"""
        return self.checkpoint_segments.copy()
    
    def is_layer_checkpointed(self, index):
        """Check if a specific layer index is checkpointed"""
        return index in self.checkpoint_segments