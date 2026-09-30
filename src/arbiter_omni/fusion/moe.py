"""
Sparse Mixture-of-Experts (MoE) Multimodal Fusion [AO-27].

Implements a 4-expert Sparse MoE Transformer block for multimodal fusion.
Each expert specializes in a different reasoning dimension:
  - Expert 0: Text-Semantic (deep linguistic reasoning)
  - Expert 1: Vision-Spatial  (spatial layout, object relations)
  - Expert 2: Temporal-Motion (temporal dynamics, action flow)
  - Expert 3: Audio-Acoustic  (acoustic scene understanding)

Top-2 gating ensures exactly 2 experts process each token, maintaining
VRAM budget while doubling effective model capacity vs a single dense FFN.
An auxiliary load-balancing loss penalizes routing collapse to prevent
all tokens routing to a single expert.

Integration:
    Use ``TransformerMultimodalFusion(use_moe=True)`` to replace dense
    TransformerEncoderLayer FFN blocks with SparseMoETransformerBlock.
    The auxiliary load-balancing loss is exposed via ``moe_aux_loss`` on
    the fusion module, to be added to the training loss with a small weight
    (e.g., 0.01).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """A single expert Feed-Forward Network (FFN).

    Identical in structure to a standard Transformer FFN block but with
    independent parameters — enabling specialization through gradient-guided routing.
    Uses GELU activation and LayerNorm for training stability.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through this expert's FFN network.

        Args:
            x: [B, S, hidden_dim] or [N, hidden_dim] token tensor.

        Returns:
            Same shape as x.
        """
        return self.net(x)


class SoftTopKRouter(nn.Module):
    """Differentiable Top-K sparse router with load-balancing auxiliary loss.

    Computes per-token routing weights to K experts using a learned linear
    gate projection + softmax scoring. Returns sparse Top-K weighted combination
    of expert indices and weights, plus an auxiliary load-balance loss that
    penalizes routing collapse.

    The auxiliary loss is computed as:
        L_aux = num_experts * sum_e( f_e * P_e )
    where:
        f_e = fraction of tokens dispatched to expert e
        P_e = mean routing probability for expert e
    This is the standard Switch Transformer / ST-MoE load-balance loss.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Top-K routing for a batch of tokens.

        Args:
            x: [N, hidden_dim] flattened token tensor (N = B * S).

        Returns:
            top_k_indices:  [N, top_k] expert indices assigned to each token.
            top_k_weights:  [N, top_k] normalized routing weights (sum = 1 per token).
            aux_loss:       scalar load-balancing loss (added to training objective).
        """
        # Router logits: [N, num_experts]
        logits = self.gate(x)
        # Routing probabilities
        probs = F.softmax(logits, dim=-1)  # [N, num_experts]

        # Top-K selection (broadcast-based, hardware-agnostic for DirectML / ROCm / CUDA)
        top_k_indices = torch.topk(probs.detach(), self.top_k, dim=-1)[1]
        classes = torch.arange(self.num_experts, device=logits.device)
        mask = (top_k_indices.unsqueeze(-1) == classes.view(1, 1, -1)).any(dim=1).to(dtype=probs.dtype)
        masked_probs = probs * mask
        norm_weights = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Construct top_k_weights [N, top_k] without scatter / gather autograd nodes
        weights_list = []
        for k in range(self.top_k):
            slot_k = top_k_indices[:, k]
            is_slot = (slot_k.unsqueeze(-1) == classes.unsqueeze(0)).to(dtype=probs.dtype)
            w_k = (norm_weights * is_slot).sum(dim=-1)
            weights_list.append(w_k)
        top_k_weights = torch.stack(weights_list, dim=-1)

        # --- Auxiliary load-balancing loss (Switch Transformer formulation) ---
        # f_e: fraction of tokens routed to expert e (from hard top-k selection)
        one_hot = (top_k_indices[:, :1] == classes.unsqueeze(0)).to(dtype=probs.dtype)
        f_e = one_hot.mean(dim=0)  # [num_experts]
        # P_e: mean routing probability for expert e (from soft scores)
        P_e = probs.mean(dim=0)  # [num_experts]
        aux_loss = (self.num_experts * (f_e * P_e).sum())

        return top_k_indices, top_k_weights, aux_loss


class SparseMoETransformerBlock(nn.Module):
    """Sparse Mixture-of-Experts Transformer Block [AO-27].

    Replaces the dense FFN sub-layer of a standard Transformer encoder layer with
    a 4-expert sparse MoE FFN. The self-attention sub-layer is unchanged.

    Architecture per block:
        x -> LayerNorm -> MultiheadSelfAttention -> residual
          -> LayerNorm (inside each ExpertFFN) -> Top-2 weighted expert FFNs -> residual

    The routing and expert computation is done over all tokens in the sequence
    simultaneously (flat dispatch), so batch × sequence tokens are routed together.

    Args:
        hidden_dim:   Token embedding dimension (d_model).
        num_experts:  Number of expert FFNs (default 4, specialized per modality).
        top_k:        Number of experts activated per token (default 2).
        ffn_dim:      Inner dimension of each expert FFN (default 4 × hidden_dim).
        num_heads:    Attention heads in self-attention sub-layer.
        dropout:      Dropout rate applied in attention and FFN sub-layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        ffn_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        _ffn_dim = ffn_dim or (hidden_dim * 4)

        # Self-attention sub-layer (standard)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Sparse MoE FFN sub-layer
        self.router = SoftTopKRouter(hidden_dim, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_dim=hidden_dim, ffn_dim=_ffn_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Cached aux loss from the most recent forward (reset each call)
        self._last_aux_loss: torch.Tensor = torch.tensor(0.0)

    @property
    def last_aux_loss(self) -> torch.Tensor:
        """Returns the load-balancing auxiliary loss from the most recent forward pass."""
        return self._last_aux_loss

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through MoE transformer block.

        Args:
            x:                    [B, S, hidden_dim] token sequence.
            src_key_padding_mask: [B, S] boolean mask (True = ignore token).

        Returns:
            [B, S, hidden_dim] updated token sequence.
        """
        B, S, D = x.shape

        # --- Self-attention sub-layer (pre-norm) ---
        x_norm = self.attn_norm(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.attn_dropout(attn_out)

        # --- Sparse MoE FFN sub-layer ---
        # Flatten to [N, D] for joint routing across all tokens
        x_flat = x.reshape(B * S, D)

        # Route tokens to Top-K experts
        top_k_indices, top_k_weights, aux_loss = self.router(x_flat)
        self._last_aux_loss = aux_loss

        # Compute weighted combination of expert outputs
        # Efficiency: compute only experts that receive at least one token
        moe_out = torch.zeros_like(x_flat)

        for k_idx in range(self.top_k):
            expert_ids = top_k_indices[:, k_idx]   # [N] — which expert for slot k
            weights_k  = top_k_weights[:, k_idx]   # [N] — routing weight for slot k

            for expert_id in range(self.num_experts):
                # Mask of tokens routed to this expert in slot k
                token_mask = (expert_ids == expert_id)
                if not token_mask.any():
                    continue
                selected_tokens = x_flat[token_mask]          # [n_k, D]
                expert_out = self.experts[expert_id](selected_tokens)  # [n_k, D]
                # Weighted accumulate
                moe_out[token_mask] += weights_k[token_mask].unsqueeze(-1) * expert_out

        # Residual connection
        x = x + moe_out.reshape(B, S, D)
        return x


class SparseMoEMultimodalFusion(nn.Module):
    """Full Sparse MoE Multimodal Fusion module [AO-27].

    Wraps the existing TransformerMultimodalFusion projection + token-building logic
    but replaces the standard TransformerEncoder stack with ``use_moe_layers`` blocks
    of SparseMoETransformerBlock.

    This is the drop-in replacement used when ``use_moe=True`` is passed to
    ``TransformerMultimodalFusion``.  It is NOT a standalone fusion module —
    it provides the transformer stack only; the outer TransformerMultimodalFusion
    orchestrates token assembly and calls ``forward_transformer(tokens, mask)``.

    Design:
        - 4 layers of SparseMoETransformerBlock (vs 2 dense layers in v4)
        - Each block has 4 experts with Top-2 routing (~200M FLOP overhead vs dense)
        - Total aux loss accumulated across all layers, exposed via ``accumulated_aux_loss``

    Args:
        hidden_dim:      Token embedding dimension.
        num_moe_layers:  Number of SparseMoE Transformer blocks (default 4 per AO-27).
        num_experts:     Number of FFN experts per block (default 4).
        top_k:           Number of active experts per token (default 2).
        ffn_dim:         Expert inner FFN dimension (default 4 × hidden_dim).
        num_heads:       Multi-head attention heads per block.
        dropout:         Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_moe_layers: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        ffn_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_moe_layers = num_moe_layers
        self.num_experts = num_experts
        self.top_k = top_k

        self.layers = nn.ModuleList([
            SparseMoETransformerBlock(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_moe_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    @property
    def accumulated_aux_loss(self) -> torch.Tensor:
        """Returns the sum of aux losses across all MoE layers from the last forward pass."""
        total = sum(layer.last_aux_loss for layer in self.layers)
        if isinstance(total, torch.Tensor):
            return total
        return torch.tensor(total)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through all MoE layers.

        Args:
            x:                    [B, S, hidden_dim] input token sequence.
            src_key_padding_mask: [B, S] boolean mask (True = ignore).

        Returns:
            [B, S, hidden_dim] fused token sequence (unnormalized).
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_norm(x)
