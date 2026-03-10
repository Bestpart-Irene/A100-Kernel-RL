"""
TRLOO-augmented GRPOTrainer — fixes the 25% gradient shrinkage from GRPO self-inclusion bias.

Dr. Kernel (arXiv 2602.05885) proves that GRPO's advantage estimation includes
the current sample in its own baseline, causing E[gradient] = (1 - 1/N) * true_gradient.
With G=4, gradients are systematically 25% too small.

Fix: scale advantages by N/(N-1) after GRPO computes them. This is the TRLOO
(Turn-level Reinforce Leave-One-Out) correction — mathematically equivalent to
computing the baseline from the other G-1 samples.

Uses vanilla TRL GRPOTrainer (no Unsloth). Unsloth's compiled GRPOTrainer is
incompatible with PEFT models loaded outside FastLanguageModel.
"""
from __future__ import annotations

from typing import Any, Union

import torch
from trl import GRPOTrainer


class TRLOOGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with TRLOO advantage correction.

    Drop-in replacement: just swap GRPOTrainer → TRLOOGRPOTrainer.

    Overrides _generate_and_score_completions (not _compute_advantages,
    which does not exist in TRL 0.24.0). Advantages are computed inline
    inside that method, so we scale them after super() returns.
    """

    def __init__(self, *args, **kwargs):
        # Pop rollout_func if passed — vanilla TRL 0.24.0 doesn't support it.
        kwargs.pop("rollout_func", None)
        super().__init__(*args, **kwargs)
        self._trloo_enabled = True

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions, score them, then apply TRLOO correction."""
        output = super()._generate_and_score_completions(inputs)

        if not self._trloo_enabled:
            return output

        advantages = output.get("advantages")
        if advantages is None:
            return output

        G = self.args.num_generations
        if G <= 1:
            return output

        # Handle NaN advantages (from backend errors / masked rewards)
        if torch.isnan(advantages).any():
            output["advantages"] = self._trloo_scale_masked(advantages, G)
        else:
            scale = G / (G - 1.0)
            output["advantages"] = advantages * scale

        return output

    def _trloo_scale_masked(self, advantages: torch.Tensor, G: int) -> torch.Tensor:
        """Apply TRLOO scaling only to finite advantage values."""
        scaled = advantages.clone()
        finite_mask = torch.isfinite(scaled)

        if finite_mask.any():
            scale = G / (G - 1.0)
            scaled[finite_mask] = scaled[finite_mask] * scale

        nan_count = int((~finite_mask).sum().item())
        if nan_count:
            print(f"[trloo] scaled finite advantages, skipped {nan_count} NaN values (G={G})")

        return scaled
