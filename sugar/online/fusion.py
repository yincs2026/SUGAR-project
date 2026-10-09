"""Uncertainty-gated fusion (Section 3.6.2).

Reliability score rho(u,t) from two complementary sources:
  (i)  the committed drift score sim(u,t)   -- staleness signal
  (ii) saturating evidence mass sat(n) = n / (n + kappa)
  rho = sigma(a * sim + b * sat(n) + c_rho)   with trainable a, b, c_rho

Content-refinement gate:
  g(u,t) = sigma(W_f [h_seq || h_prof || rho] + b_f)
  h = (1 - g) * h_seq + g * h_prof

Ablation switches (batch 12):
  fixed_gate   : rho and g replaced by a constant blend weight (A3)
  no_profile   : h = h_seq always (A1)
  no_reliability: g computed without rho input (A7-style variant)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGatedFusion(nn.Module):
    def __init__(self, d: int, kappa: float = 10.0, mode: str = "full"):
        super().__init__()
        assert mode in ("full", "fixed_gate", "no_reliability")
        self.kappa = kappa
        self.mode = mode

        # a, b initialized positive; c_rho free bias (learned prior)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.c_rho = nn.Parameter(torch.tensor(0.0))

        if mode == "fixed_gate":
            # single constant blend weight, learned end-to-end
            self.fixed_weight = nn.Parameter(torch.tensor(0.5))
        elif mode == "no_reliability":
            self.W_f = nn.Linear(2 * d, d)
        else:
            # rho enters as an appended scalar -> 2d + 1 inputs
            self.W_f = nn.Linear(2 * d + 1, d)

    def reliability(self, sim: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """rho(u,t) in (0,1). sim: (B,) or (B,1); n: consolidated counts.

        Cold-start regime (sim=0, n=0) reduces to the learned prior
        sigma(c_rho), exactly as specified in Section 3.6.2.
        """
        sat = n.float() / (n.float() + self.kappa)
        return torch.sigmoid(self.a * sim.float() + self.b * sat + self.c_rho)

    def forward(self, h_seq: torch.Tensor, h_prof: torch.Tensor,
                sim: torch.Tensor, n: torch.Tensor) -> tuple:
        """Returns (h_fused, g). All tensors batch-major: (B, L, d) / (B, L).

        sim, n are per-batch (piecewise-constant across chunk positions),
        broadcast over the sequence dimension.
        """
        if self.mode == "fixed_gate":
            w = torch.clamp(self.fixed_weight, 0.0, 1.0)
            g = w.expand_as(h_seq[..., :1]).to(h_seq.dtype)
            h = (1 - g) * h_seq + g * h_prof
            return h, g.squeeze(-1)

        B, L, _ = h_seq.shape
        sim_b = sim.float().view(B, 1).expand(B, L)
        n_b = n.float().view(B, 1).expand(B, L)

        if self.mode == "no_reliability":
            rho = None
            gate_in = torch.cat([h_seq, h_prof], dim=-1)
        else:
            rho = self.reliability(sim_b, n_b).unsqueeze(-1)  # (B, L, 1)
            gate_in = torch.cat([h_seq, h_prof, rho], dim=-1)

        g = torch.sigmoid(self.W_f(gate_in))  # (B, L, d)
        h = (1 - g) * h_seq + g * h_prof
        return h, g.squeeze(-1) if g.dim() == 3 else g