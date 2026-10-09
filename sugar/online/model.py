"""SUGARModel: full assembly of the online stage (M4-M6, Algorithm 2).

Forward pass (per batch, chunked by visible snapshot):
  1. h_seq = Backbone(X_u + E_pos)[t]                      (line 3)
  2. per chunk: p~_u = SocialGate(p_u, s_u, e_bar(R_u))     (lines 5-9)
     cold-start: p_u <- p0, sim <- 0, n <- 0                (line 6)
  3. h_prof = W_p p~_u + b_p                                (line 11)
  4. rho = sigma(a sim + b sat(n) + c_rho)                  (line 12)
  5. g = sigma(W_f [h_seq || h_prof || rho] + b_f)          (line 13)
  6. h = (1 - g) h_seq + g h_prof                           (line 14)

Trainable parameters (Theta): item embeddings, Transformer blocks, W_p/b_p,
fusion scalars (a, b, c_rho) and gate (W_f, b_f), default vector p0, social
gate (w_gamma, b_gamma, beta). The frozen LLM/encoder receive NO gradient.

Ablation modes (constructor switches):
  mode="full" | "fixed_gate" (A3) | "no_reliability" | "no_profile" (A1)
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from sugar.online.backbone import TransformerBackbone
from sugar.online.projection import ProfileProjection
from sugar.online.fusion import UncertaintyGatedFusion
from sugar.online.social_gate import SocialGate


class SUGARModel(nn.Module):
    def __init__(self, n_items: int, d: int = 64, d_e: int = 1024,
                 n_blocks: int = 2, n_heads: int = 2, max_len: int = 50,
                 dropout: float = 0.2, kappa: float = 10.0,
                 max_neighbors: int = 10, mode: str = "full",
                 pad_idx: int = 0):
        super().__init__()
        self.mode = mode
        self.d_e = d_e
        self.pad_idx = pad_idx
        self.max_neighbors = max_neighbors

        self.backbone = TransformerBackbone(
            n_items, d, n_blocks, n_heads, max_len, dropout, pad_idx)
        self.projection = ProfileProjection(d_e, d)
        self.fusion = UncertaintyGatedFusion(d, kappa, mode="full"
                                             if mode == "full" else
                                             ("fixed_gate" if mode == "fixed_gate"
                                              else "no_reliability"))
        self.social_gate = SocialGate(d_e, max_neighbors)

        # learnable default profile vector for cold-start users (Section 3.2)
        self.p0 = nn.Parameter(torch.randn(d_e) * 0.02)

    # ---- social aggregation per user (M4) -----------------------------------
    def _social_signal(self, user_idx: int, window_mean: torch.Tensor,
                       neighbor_profiles: dict) -> torch.Tensor | None:
        """s_u from the latest visible neighbor snapshots (Section 3.5).

        neighbor_profiles: {neighbor_idx: np.ndarray (d_e,)} precomputed by
        the trainer from the snapshot store; only cached frozen vectors
        enter here -- no LLM, no content generation.
        """
        nbs = neighbor_profiles.get(user_idx, [])
        if not nbs:
            return None
        tensors = [torch.as_tensor(np.asarray(p), dtype=torch.float32,
                                   device=window_mean.device)
                   for p in nbs[: self.max_neighbors * 4]]
        s_u, _ = self.social_gate.aggregate(window_mean, tensors)
        return s_u

    # ---- main forward --------------------------------------------------------
    
    def forward(
        self,
        seq: torch.Tensor,
        user_indices: torch.Tensor = None,
        seq_lens: torch.Tensor = None,
        snapshot_store=None,
        neighbor_map: dict = None,
        strict: bool = True,
        window_means: dict = None,
    ) -> torch.Tensor:
        """
        Return contextual/fused sequence representations of shape (B, L, d).

        Args:
            seq: Padded item IDs, shape (B, L).
            user_indices: User ID for each row, shape (B,).
            seq_lens: True sequence lengths, shape (B,).
            snapshot_store: Committed user-profile snapshots.
            neighbor_map: Optional cached neighbor profiles.
            strict: Enforce snapshot visibility constraints.
            window_means: Optional per-user recent-window embeddings.

        Returns:
            Sequence representations with shape (B, L, d).
        """
        if not isinstance(seq, torch.Tensor) or seq.ndim != 2:
            raise ValueError("seq must be a tensor with shape (B, L).")

        # Sequence-only ablation does not require user/profile side inputs.
        if self.mode == "no_profile":
            return self.backbone(seq)

        if user_indices is None:
            raise ValueError(
                "user_indices is required for profile-aware SUGAR forward."
            )
        if seq_lens is None:
            raise ValueError(
                "seq_lens is required for profile-aware SUGAR forward."
            )
        if snapshot_store is None:
            raise ValueError(
                "snapshot_store is required for profile-aware SUGAR forward."
            )

        B, L = seq.shape
        if user_indices.ndim != 1 or user_indices.shape[0] != B:
            raise ValueError("user_indices must have shape (B,).")
        if seq_lens.ndim != 1 or seq_lens.shape[0] != B:
            raise ValueError("seq_lens must have shape (B,).")

        return self.forward_with_users(
            seq=seq,
            user_indices=user_indices,
            seq_lens=seq_lens,
            snapshot_store=snapshot_store,
            neighbor_map=neighbor_map,
            window_means=window_means,
            strict=strict,
        )


    def forward_with_users(self, seq: torch.Tensor, user_indices: torch.Tensor,
                           seq_lens: torch.Tensor, snapshot_store,
                           neighbor_map: dict = None, window_means: dict = None,
                           strict: bool = True) -> torch.Tensor:
        """Full fused forward. user_indices: (B,) integer user ids aligned
        with seq rows; window_means: {user_idx: (d_e,) recent-window mean}."""
        B, L = seq.shape
        device = seq.device
        h_seq = self.backbone(seq)

        if self.mode == "no_profile":
            return h_seq

        h_out = h_seq.clone()
        sim_all = torch.zeros(B, device=device)
        n_all = torch.zeros(B, device=device)

        for b in range(B):
            u = int(user_indices[b].item())
            Lb = int(seq_lens[b].item())
            view_snaps = snapshot_store.snapshots_of(u)
            strict_snaps = [s for s in view_snaps if s["t_commit"] <= Lb - 1] \
                if strict else view_snaps
            info = strict_snaps[-1] if strict_snaps else None

            # cold-start substitution (Algorithm 2, line 6)
            if info is None or info["profile_emb"] is None:
                p_u = self.p0
                sim, n = 0.0, 0.0
            else:
                p_u = torch.as_tensor(np.asarray(info["profile_emb"]),
                                      dtype=torch.float32, device=device)
                sim, n = float(info["sim"]), float(info["n_consolidated"])

            wm = window_means.get(u) if window_means else None
            if wm is not None:
                wm_t = torch.as_tensor(np.asarray(wm), dtype=torch.float32,
                                       device=device)
            else:
                wm_t = torch.zeros(self.d_e, device=device)

            # M4: socially-grounded profile (skipped in no-social ablations
            # by passing neighbor_map=None)
            s_u = None
            if neighbor_map:
                s_u = self._social_signal(u, wm_t, neighbor_map)
            p_tilde, _ = self.social_gate.blend(p_u, s_u, wm_t)

            h_prof = self.projection(p_tilde)                     # (d,)

            sim_all[b] = sim
            n_all[b] = n

            # fusion across this user's whole window (piecewise-constant
            # side info within the chunk; chunk-level processing comes in
            # the trainer for memory efficiency)
            g = self._gate_for(h_seq[b], h_prof, sim, n)          # (L, d)
            h_out[b] = (1 - g) * h_seq[b] + g * h_prof.unsqueeze(0).expand(
                L, -1)

        return h_out

    def _gate_for(self, h_row: torch.Tensor, h_prof: torch.Tensor,
                  sim: float, n: float) -> torch.Tensor:
        """Per-position fusion gate for one user. h_row: (L, d)."""
        if self.fusion.mode == "fixed_gate":
            w = torch.clamp(self.fusion.fixed_weight, 0.0, 1.0)
            return w.expand(h_row.size(0), 1).to(h_row.dtype)
        L = h_row.size(0)
        sim_t = torch.tensor(sim, device=h_row.device)
        n_t = torch.tensor(n, device=h_row.device)
        rho = self.fusion.reliability(sim_t, n_t)                 # scalar
        if self.fusion.mode == "no_reliability":
            gate_in = torch.cat([h_row, h_prof.unsqueeze(0).expand(L, -1)],
                                dim=-1)
        else:
            rho_col = rho.expand(L, 1)
            gate_in = torch.cat([h_row, h_prof.unsqueeze(0).expand(L, -1),
                                 rho_col], dim=-1)
        return torch.sigmoid(self.fusion.W_f(gate_in))

    # ---- scoring --------------------------------------------------------------
    def score_items(self, h: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """y(u,v) = h^T x_v for candidate items. h: (..., d) -> (..., n_cand)."""
        x = self.backbone.embed_items(item_indices)               # (n_cand, d)
        return h @ x.T

    def all_item_embeddings(self) -> torch.Tensor:
        return self.backbone.item_emb.weight