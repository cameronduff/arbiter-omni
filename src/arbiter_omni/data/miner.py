"""
Hard-Negative Candidate Mining using semantic cosine similarity on frozen representations.
Hardens the dynamic decision head against adversarial foils and near-identical alternatives.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import torch
import torch.nn.functional as F

from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.types import MultimodalSample


# Default open-domain candidate pool across diverse tasks
DEFAULT_CANDIDATE_POOL = [
    # Navigation & Robotics
    "proceed at nominal velocity",
    "proceed with caution",
    "decelerate immediately",
    "execute emergency stop",
    "maintain current trajectory",
    "steer left to avoid obstruction",
    "steer right to avoid obstruction",
    "reverse throttle",
    "adjust altitude upward",
    "adjust altitude downward",
    "hover at current position",
    "dock with charging station",
    "engage mechanical gripper",
    "release mechanical gripper",
    "recalibrate sensor orientation",
    "request operator teleoperation",
    # Science & Reasoning
    "increase system temperature",
    "decrease system temperature",
    "increase pressure threshold",
    "decrease pressure threshold",
    "add catalyst reagent",
    "dilute solution with solvent",
    "record baseline measurement",
    "purge reaction chamber",
    # Quality & Inspection
    "accept component as compliant",
    "flag minor surface defect",
    "reject component due to critical flaw",
    "schedule preventive maintenance",
    "reroute item to secondary inspection",
    # General Decision
    "confirm transaction",
    "reject transaction as anomalous",
    "defer decision pending further evidence",
    "escalate to human supervisor",
    "log event and continue",
]


class HardNegativeMiner:
    """
    Mines semantically adjacent candidate foils using text encoder cosine similarity.
    
    Given a target candidate choice, retrieves candidate foils from a candidate pool
    that are semantically close (high cosine similarity) without being exact duplicates,
    forcing the decision head to learn sharp, calibrated discrimination boundaries.
    """

    def __init__(
        self,
        encoder: BaseMultimodalEncoder,
        candidate_pool: Optional[Sequence[str]] = None,
        batch_size: int = 64,
        device: Optional[Union[str, torch.device]] = None,
        memory_bank: Optional[Any] = None,
    ):
        self.encoder = encoder
        self.device = device or getattr(encoder, "device", torch.device("cpu"))
        self.memory_bank = memory_bank
        
        # Deduplicate pool
        pool_raw = candidate_pool if candidate_pool is not None else DEFAULT_CANDIDATE_POOL
        seen = set()
        self.pool: List[str] = []
        for c in pool_raw:
            c_str = str(c).strip()
            if c_str and c_str not in seen:
                seen.add(c_str)
                self.pool.append(c_str)

        if not self.pool:
            raise ValueError("Candidate pool cannot be empty.")

        # Pre-compute L2-normalized embeddings for the candidate pool
        self.pool_embeddings = self._encode_texts(self.pool, batch_size=batch_size)

        if self.memory_bank is not None:
            self.populate_memory_bank(self.memory_bank)

    def populate_memory_bank(self, bank: Optional[Any] = None) -> int:
        """Populates a PersistentMemoryBank with candidate pool representations."""
        target_bank = bank if bank is not None else self.memory_bank
        if target_bank is None:
            return 0
        return target_bank.enqueue(self.pool_embeddings, texts=self.pool)

    def _encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        """Encodes texts into L2-normalized vectors."""
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            with torch.no_grad():
                emb = self.encoder.encode_text(chunk)
                # Ensure L2 normalized
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                all_embeds.append(emb.cpu())
        return torch.cat(all_embeds, dim=0)  # [M, D] on CPU

    @classmethod
    def from_dataset(
        cls,
        dataset: Sequence[MultimodalSample],
        encoder: BaseMultimodalEncoder,
        batch_size: int = 64,
        additional_candidates: Optional[Sequence[str]] = None,
    ) -> HardNegativeMiner:
        """Constructs miner by harvesting all candidate choices present across a dataset."""
        candidates = set()
        for sample in dataset:
            if hasattr(sample, "candidates"):
                for c in sample.candidates:
                    candidates.add(str(c).strip())
        if additional_candidates:
            for c in additional_candidates:
                candidates.add(str(c).strip())
        
        return cls(encoder=encoder, candidate_pool=sorted(list(candidates)), batch_size=batch_size)

    def mine_hard_negatives(
        self,
        target_text: str,
        k: int = 3,
        min_similarity: float = 0.25,
        max_similarity: float = 0.98,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Finds the top-k nearest semantic candidates in the pool for a given target candidate.
        
        Args:
            target_text: The ground-truth candidate or reference text.
            k: Maximum number of hard negative foils to return.
            min_similarity: Minimum cosine similarity threshold (rejects completely irrelevant choices).
            max_similarity: Maximum cosine similarity threshold (rejects exact duplicates/synonyms).
            exclude: Optional list of texts to strictly exclude (e.g. existing candidates in a sample).
            
        Returns:
            List of mined candidate strings, ranked descending by similarity.
        """
        exclude_set = {str(target_text).strip().lower()}
        if exclude:
            for ex in exclude:
                exclude_set.add(str(ex).strip().lower())

        with torch.no_grad():
            target_emb = self.encoder.encode_text([target_text]).cpu()
            target_emb = target_emb / (target_emb.norm(dim=-1, keepdim=True) + 1e-8)  # [1, D]

        # Cosine similarities: [1, M]
        sims = (target_emb @ self.pool_embeddings.T).squeeze(0)  # [M]
        sorted_indices = torch.argsort(sims, descending=True)

        mined: List[str] = []
        for idx in sorted_indices.tolist():
            sim = float(sims[idx].item())
            cand_text = self.pool[idx]
            
            if cand_text.strip().lower() in exclude_set:
                continue
            if sim > max_similarity:
                continue
            if sim < min_similarity:
                # If even the best remaining candidate is below min_similarity, break
                break

            mined.append(cand_text)
            if len(mined) >= k:
                break

        return mined

    def augment_sample(
        self,
        sample: MultimodalSample,
        num_hard_negatives: int = 1,
        min_similarity: float = 0.25,
        max_similarity: float = 0.98,
    ) -> MultimodalSample:
        """
        Augments a MultimodalSample by mining hard negatives for its target candidate
        and adding them to sample.candidates.
        """
        if sample.target_idx is None or sample.target_idx >= len(sample.candidates):
            return sample

        target_text = sample.candidates[sample.target_idx]
        hard_negs = self.mine_hard_negatives(
            target_text=target_text,
            k=num_hard_negatives,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            exclude=sample.candidates,
        )

        if not hard_negs:
            return sample

        new_candidates = list(sample.candidates) + hard_negs
        # target_idx stays the same since we append
        new_sample = MultimodalSample(
            question=sample.question,
            candidates=new_candidates,
            target_idx=sample.target_idx,
            text=sample.text,
            image=sample.image,
            video=sample.video,
            audio=sample.audio,
            metadata={
                **(sample.metadata or {}),
                "mined_hard_negatives": hard_negs,
            },
        )
        return new_sample

    def augment_dataset(
        self,
        samples: Sequence[MultimodalSample],
        num_hard_negatives: int = 1,
        min_similarity: float = 0.25,
        max_similarity: float = 0.98,
    ) -> List[MultimodalSample]:
        """Augments every sample in a list with mined hard negative candidate foils."""
        augmented = []
        for s in samples:
            augmented.append(
                self.augment_sample(
                    s,
                    num_hard_negatives=num_hard_negatives,
                    min_similarity=min_similarity,
                    max_similarity=max_similarity,
                )
            )
        return augmented
