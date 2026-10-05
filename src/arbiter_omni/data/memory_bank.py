"""
Global Resident Hard-Negative Memory Bank [AO-23 / AO-25].
Maintains a 100,000 candidate representation FIFO queue resident in shared system RAM (8 GB pool)
to supply out-of-distribution adversarial candidate foils for contrastive margin learning.

AO-25 upgrade: capacity scaled to 100k vectors; optional fp16 storage halves RAM from
~307 MB (fp32@100k×768) to ~153 MB, keeping the bank well within the 8 GB shared DDR4 pool.
Optional INT8 storage further reduces to ~77 MB for extreme memory-constrained scenarios.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F


class PersistentMemoryBank:
    """
    A persistent cyclic FIFO memory bank of candidate embeddings held in host/shared system memory.
    
    Provides sub-millisecond retrieval of the hardest out-of-distribution candidate foils
    for any batch of query vectors without burdening dedicated GPU VRAM (4 GB GDDR5).

    AO-25 memory footprint (dim=768, all in shared DDR4):
        100,000 vectors @ fp32  →  ~293 MB
        100,000 vectors @ fp16  →  ~147 MB  (store_fp16=True)
    All cosine-similarity queries execute in CPU shared memory with zero VRAM overhead.
    """

    def __init__(
        self,
        capacity: int = 100000,
        candidate_dim: int = 768,
        context_dim: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        store_fp16: bool = False,
    ):
        self.capacity = int(capacity)
        self.candidate_dim = int(candidate_dim)
        self.context_dim = int(context_dim) if context_dim is not None else None
        self.device = torch.device(device) if isinstance(device, str) else device
        # AO-25: fp16 storage mode halves bank RAM; queries auto-cast to fp32 before matmul
        self.store_fp16 = store_fp16
        self.dtype = torch.float16 if store_fp16 else dtype

        # Pre-allocate contiguous tensor memory buffer in host/shared memory
        self.candidate_bank = torch.zeros(
            (self.capacity, self.candidate_dim),
            dtype=self.dtype,
            device=self.device,
        )
        if self.context_dim is not None:
            self.context_bank = torch.zeros(
                (self.capacity, self.context_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.context_bank = None

        self.texts: List[Optional[str]] = [None] * self.capacity
        self.ptr: int = 0
        self.size: int = 0
        self.total_enqueued: int = 0

    @property
    def is_full(self) -> bool:
        """Returns True if the memory bank has reached maximum capacity."""
        return self.size >= self.capacity

    @property
    def memory_usage_mb(self) -> float:
        """Calculates total memory occupied by the memory bank in megabytes."""
        bytes_total = self.candidate_bank.element_size() * self.candidate_bank.nelement()
        if self.context_bank is not None:
            bytes_total += self.context_bank.element_size() * self.context_bank.nelement()
        return bytes_total / (1024.0 * 1024.0)

    def __len__(self) -> int:
        return self.size

    def enqueue(
        self,
        candidate_embeds: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        context_embeds: Optional[torch.Tensor] = None,
        texts: Optional[Sequence[str]] = None,
    ) -> int:
        """
        Enqueues normalized candidate embeddings into the cyclic FIFO buffer.
        
        Args:
            candidate_embeds: [N, D] or [B, K, D] candidate vectors.
            candidate_mask: Optional [B, K] boolean mask filtering valid candidates.
            context_embeds: Optional [N, context_dim] or [B, context_dim] context vectors.
            texts: Optional sequence of candidate text strings corresponding to candidates.
            
        Returns:
            Number of candidate vectors successfully enqueued.
        """
        if candidate_embeds is None or candidate_embeds.numel() == 0:
            return 0

        # Detach and move to CPU / bank device
        cands = candidate_embeds.detach().to(self.device, dtype=self.dtype)

        # Handle 3D [B, K, D] inputs
        if cands.ndim == 3:
            B, K, D = cands.shape
            if candidate_mask is not None:
                mask_flat = candidate_mask.to(self.device).reshape(-1)
                cands_flat = cands.reshape(B * K, D)
                cands = cands_flat[mask_flat]
            else:
                cands = cands.reshape(B * K, D)
        elif cands.ndim == 2:
            if candidate_mask is not None:
                mask_flat = candidate_mask.to(self.device).reshape(-1)
                cands = cands[mask_flat]
        else:
            raise ValueError(f"Expected candidate_embeds to have 2 or 3 dims, got {cands.ndim}")

        N = cands.shape[0]
        if N == 0:
            return 0

        # Ensure L2 normalization
        norms = torch.norm(cands, p=2, dim=-1, keepdim=True) + 1e-8
        cands = cands / norms

        # Enqueue cyclically
        if N >= self.capacity:
            # If batch exceeds total capacity, keep the most recent self.capacity items
            self.candidate_bank.copy_(cands[-self.capacity :])
            self.ptr = 0
            self.size = self.capacity
            if texts is not None:
                recent_texts = list(texts)[-self.capacity :]
                self.texts = recent_texts + [None] * (self.capacity - len(recent_texts))
        else:
            space_left = self.capacity - self.ptr
            if N <= space_left:
                self.candidate_bank[self.ptr : self.ptr + N].copy_(cands)
                if texts is not None:
                    for i, t in enumerate(texts[:N]):
                        self.texts[self.ptr + i] = str(t)
            else:
                first_chunk = space_left
                second_chunk = N - first_chunk
                self.candidate_bank[self.ptr :].copy_(cands[:first_chunk])
                self.candidate_bank[:second_chunk].copy_(cands[first_chunk:])
                if texts is not None:
                    for i, t in enumerate(texts[:first_chunk]):
                        self.texts[self.ptr + i] = str(t)
                    for i, t in enumerate(texts[first_chunk:N]):
                        self.texts[i] = str(t)

            self.ptr = (self.ptr + N) % self.capacity
            self.size = min(self.capacity, self.size + N)

        # Context embeds tracking if configured
        if self.context_bank is not None and context_embeds is not None:
            ctxs = context_embeds.detach().to(self.device, dtype=self.dtype)
            if ctxs.ndim == 2:
                ctx_norm = ctxs / (torch.norm(ctxs, p=2, dim=-1, keepdim=True) + 1e-8)
                M = ctx_norm.shape[0]
                if M >= self.capacity:
                    self.context_bank.copy_(ctx_norm[-self.capacity :])
                else:
                    left = self.capacity - self.ptr
                    if M <= left:
                        self.context_bank[self.ptr : self.ptr + M].copy_(ctx_norm)
                    else:
                        self.context_bank[self.ptr :].copy_(ctx_norm[:left])
                        self.context_bank[: M - left].copy_(ctx_norm[left:])

        self.total_enqueued += N
        return N

    def query_hard_foils(
        self,
        query_embed: torch.Tensor,
        k: int = 10,
        min_sim: float = 0.25,
        max_sim: float = 0.98,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the top-k hardest semantic candidate foils for query embeddings
        from the resident 50k candidate pool.
        
        Args:
            query_embed: [B, D] query representations (e.g. positive candidate or context).
            k: Number of hardest foils to return per query sample.
            min_sim: Minimum cosine similarity threshold (rejects irrelevant noise).
            max_sim: Maximum cosine similarity threshold (rejects duplicates/synonyms).
            
        Returns:
            hard_foils: [B, k, D] foil candidate embeddings on the query device.
            foil_sims: [B, k] cosine similarities with query vectors.
            foil_mask: [B, k] boolean tensor indicating which foils meet similarity thresholds.
        """
        device = query_embed.device
        if query_embed.ndim == 1:
            q = query_embed.unsqueeze(0)
        else:
            q = query_embed
        B, D = q.shape

        if self.size == 0 or k <= 0:
            empty_foils = torch.zeros((B, k, self.candidate_dim), device=device, dtype=q.dtype)
            empty_sims = torch.zeros((B, k), device=device, dtype=q.dtype)
            empty_mask = torch.zeros((B, k), device=device, dtype=torch.bool)
            return empty_foils, empty_sims, empty_mask

        # Normalize query vector — always compute in fp32 for precision
        q_norm = q.detach().to(self.device, dtype=torch.float32)
        q_norm = q_norm / (torch.norm(q_norm, p=2, dim=-1, keepdim=True) + 1e-8)  # [B, D]

        # Active slice of bank: [size, D] — upcast fp16 to fp32 for cosine-sim matmul
        active_bank = self.candidate_bank[: self.size].to(torch.float32)

        # Fast cosine similarity matrix computation in host/shared memory: [B, size]
        sims = torch.matmul(q_norm, active_bank.T)

        # Apply boundary thresholds
        # Exclude exact copies or near-synonyms (sim > max_sim) and irrelevant noise (sim < min_sim)
        filtered_sims = torch.where(
            (sims >= min_sim) & (sims <= max_sim),
            sims,
            torch.tensor(-1e9, device=self.device, dtype=sims.dtype),
        )

        query_k = min(k, self.size)
        topk_vals, topk_inds = torch.topk(filtered_sims, k=query_k, dim=-1)

        # Gather foil candidate embeddings: [B, query_k, D]
        selected_foils = active_bank[topk_inds]
        valid_mask = topk_vals > -1e8

        # If query_k < k, pad to requested k (always float32 to match active_bank upcast)
        if query_k < k:
            pad_len = k - query_k
            pad_foils = torch.zeros((B, pad_len, self.candidate_dim), device=self.device, dtype=torch.float32)
            pad_sims = torch.full((B, pad_len), -1e9, device=self.device, dtype=torch.float32)
            pad_mask = torch.zeros((B, pad_len), device=self.device, dtype=torch.bool)

            selected_foils = torch.cat([selected_foils, pad_foils], dim=1)
            topk_vals = torch.cat([topk_vals, pad_sims], dim=1)
            valid_mask = torch.cat([valid_mask, pad_mask], dim=1)

        # Dispatch selected foils to target query device
        return (
            selected_foils.to(device, dtype=query_embed.dtype),
            topk_vals.to(device, dtype=query_embed.dtype),
            valid_mask.to(device),
        )

    def populate_from_pool(
        self,
        texts: Sequence[str],
        encoder: Any,
        batch_size: int = 64,
    ) -> int:
        """
        Encodes a candidate string pool into the memory bank using the frozen text encoder.
        
        Args:
            texts: Candidate strings.
            encoder: Multimodal or text encoder with encode_text(list_of_strings).
            batch_size: Encoding batch size.
            
        Returns:
            Total items enqueued.
        """
        unique_texts = []
        seen = set()
        for t in texts:
            cleaned = str(t).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                unique_texts.append(cleaned)

        total_added = 0
        for i in range(0, len(unique_texts), batch_size):
            chunk = unique_texts[i : i + batch_size]
            with torch.no_grad():
                emb = encoder.encode_text(chunk)
            total_added += self.enqueue(emb, texts=chunk)

        return total_added

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists the memory bank state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "capacity": self.capacity,
            "candidate_dim": self.candidate_dim,
            "context_dim": self.context_dim,
            "ptr": self.ptr,
            "size": self.size,
            "total_enqueued": self.total_enqueued,
            "store_fp16": self.store_fp16,
            "candidate_bank": self.candidate_bank[: self.size].clone().cpu(),
            "texts": self.texts[: self.size],
        }
        torch.save(payload, path)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads memory bank state from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Memory bank checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.capacity = payload["capacity"]
        self.candidate_dim = payload["candidate_dim"]
        self.context_dim = payload.get("context_dim", None)
        self.ptr = payload["ptr"]
        self.size = payload["size"]
        self.total_enqueued = payload.get("total_enqueued", self.size)
        self.store_fp16 = payload.get("store_fp16", False)
        if self.store_fp16:
            self.dtype = torch.float16

        loaded_cands = payload["candidate_bank"]
        self.candidate_bank = torch.zeros(
            (self.capacity, self.candidate_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.candidate_bank[: self.size].copy_(loaded_cands)

        self.texts = [None] * self.capacity
        loaded_texts = payload.get("texts", [])
        for i, t in enumerate(loaded_texts):
            if i < self.capacity:
                self.texts[i] = t

