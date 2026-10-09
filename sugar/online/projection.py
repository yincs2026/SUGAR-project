"""Trainable projection from the frozen semantic space to the backbone space
(Section 3.6.1): h_prof = W_p p~_u + b_p, with p~ in R^{d_e}, h_prof in R^d.

The SAME projection is reused by the alignment loss for item embeddings
(W_p e_v in R^d, Section 3.6.3), coupling item-level and profile-level
translators into one mapping with no extra parameters.
"""
import torch
import torch.nn as nn


class ProfileProjection(nn.Module):
    def __init__(self, d_e: int, d: int):
        super().__init__()
        self.W_p = nn.Linear(d_e, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_e) frozen-space vector(s) -> (..., d) backbone space."""
        return self.W_p(x)