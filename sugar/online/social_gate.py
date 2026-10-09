"""M4: Social Gate and Aggregator (Section 3.5).

Affinity: alpha_uw = softmax_w(cos(e_bar(R_u), p_w) / beta) over neighbors.
Neighborhood vector: s_u = sum_w alpha_uw p_w over the top-L neighbors.
Gate: gamma_u = sigma(w_gamma^T [p_u || s_u || e_bar(R_u)] + b_gamma).
Output: p~_u = (1 - gamma_u) p_u + gamma_u s_u.

Computed inside the training graph from FIXED snapshots (piecewise-constant
per checkpoint interval, O(L * d_e)); gradients flow only into
w_gamma, b_gamma, beta -- the profile content is never modified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialGate(nn.Module):
    def __init__(self, d_e: int, max_neighbors: int = 10):
        super().__init__()
        self.d_e = d_e
        self.L = max_neighbors
        self.w_gamma = nn.Parameter(torch.randn(3 * d_e) * 0.01)
        self.b_gamma = nn.Parameter(torch.zeros(1))
        # learnable temperature beta > 0, parameterized via softplus
        self._beta_raw = nn.Parameter(torch.tensor(1.0))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw) + 1e-4

    def aggregate(self, user_window_mean: torch.Tensor,
                  neighbor_profiles: list) -> tuple:
        """Compute s_u for one user from candidate neighbor profile vectors.

        user_window_mean: (d_e,) e_bar(R_u)
        neighbor_profiles: list of (d_e,) tensors, frozen snapshot vectors
        Returns (s_u (d_e,), affinity list) -- empty list if no neighbors.
        """
        if not neighbor_profiles:
            return torch.zeros_like(user_window_mean), []
        P = torch.stack(neighbor_profiles)                    # (M, d_e)
        sims = P @ user_window_mean                           # (M,)
        # top-L selection before renormalization (Section 3.5)
        M = sims.size(0)
        if M > self.L:
            top = torch.topk(sims, self.L)
            P, sims = P[top.indices], top.values
        alpha = F.softmax(sims / self.beta, dim=0)            # renormalized
        s_u = (alpha.unsqueeze(-1) * P).sum(dim=0)
        return s_u, alpha.detach().tolist()

    def gate(self, p_u: torch.Tensor, s_u: torch.Tensor,
             window_mean: torch.Tensor) -> torch.Tensor:
        """gamma_u in (0,1) for one user (Section 3.5)."""
        z = torch.cat([p_u, s_u, window_mean])
        return torch.sigmoid(self.w_gamma @ z + self.b_gamma)

    def blend(self, p_u: torch.Tensor, s_u: torch.Tensor,
              window_mean: torch.Tensor) -> tuple:
        """Full M4 output: socially-grounded profile p~_u.

        Users with no social signal return (p_u, gamma=0) unchanged.
        Cold-start users pass p_u = p0 so the gate can open on social
        evidence alone (Section 3.5).
        """
        if s_u is None or s_u.abs().sum() == 0:
            return p_u, torch.zeros(1, device=p_u.device)
        gamma = self.gate(p_u, s_u, window_mean)
        p_tilde = (1 - gamma) * p_u + gamma * s_u
        return p_tilde, gamma

    def forward(self, p_u: torch.Tensor, s_u: torch.Tensor,
                window_mean: torch.Tensor) -> torch.Tensor:
        p_tilde, _ = self.blend(p_u, s_u, window_mean)
        return p_tilde