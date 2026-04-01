# -*- coding: utf-8 -*-
"""
Diffractive Deep Neural Network (ONN) - dynamic number of layers
Supports any number of phase-modulating layers by changing z_list length
"""

import torch
from torch import nn
from OpticalLayers import DiffLayer, Diffraction


class Onn(nn.Module):
    def __init__(self, M, L, lambda0, z_list):
        """
        z_list: list or tuple of propagation distances
                len(z_list) = num_phase_layers + 1
        
        Example:
            z_list = [z_in, z1, z2, z3, ..., zN]
            → 1 input diffraction + N phase layers + N propagations
        """
        super(Onn, self).__init__()
        
        if len(z_list) < 2:
            raise ValueError("z_list must contain at least input propagation + one more distance")
        
        self.layers = nn.ModuleList()
        
        # First free-space propagation (no trainable phase mask)
        self.layers.append(Diffraction(M, L, lambda0, z_list[0]))
        
        # Then: phase modulation → propagation, repeated for each remaining distance
        for z in z_list[1:]:
            self.layers.append(DiffLayer(M, L, lambda0, z))

    def forward(self, u):
        for layer in self.layers:
            u = layer(u)
        return u

    def get_phase_masks(self):
        """
        Returns list of trainable phase masks (as numpy arrays) from all DiffLayer modules.
        Used by visualization and export scripts.
        """
        return [
            module.params.detach().cpu().squeeze(0).numpy()
            for module in self.layers
            if isinstance(module, DiffLayer)
        ]

    def get_num_phase_layers(self):
        """Convenience method"""
        return sum(1 for m in self.layers if isinstance(m, DiffLayer))