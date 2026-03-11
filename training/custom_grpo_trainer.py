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

import os
from typing import Any, Union

import torch
from trl import GRPOTrainer


class TRLOOGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with TRLOO advantage correction and optional OVF filtering.

    Drop-in replacement: just swap GRPOTrainer → TRLOOGRPOTrainer.

    Overrides _generate_and_score_completions (not _compute_advantages,
    which does not exist in TRL 0.24.0). Advantages are computed inline
    inside that method, so we scale them after super() returns.

    OVF (Optimal Variance Filtering): After generating G completions per prompt,
    only includes the highest-variance groups in the GRPO update. G=8 with OVF
    can match G=16 without it by focusing learning on informative samples.
    Enable with KERNELFORGE_OVF=1, set threshold with KERNELFORGE_OVF_TOP_FRAC.
    """

    def __init__(self, *args, **kwargs):
        # Pop rollout_func if passed — vanilla TRL 0.24.0 doesn't support it.
        kwargs.pop("rollout_func", None)
        super().__init__(*args, **kwargs)
        self._trloo_enabled = True
        self._ovf_enabled = os.getenv("KERNELFORGE_OVF", "0") == "1"
        # Fraction of prompt groups to keep (by reward variance). Default: top 50%.
        self._ovf_top_frac = float(os.getenv("KERNELFORGE_OVF_TOP_FRAC", "0.5"))

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions, score them, then apply TRLOO correction + OVF."""
        output = super()._generate_and_score_completions(inputs)

        advantages = output.get("advantages")
        if advantages is None:
            return output

        G = self.args.num_generations
        if G <= 1:
            return output

        # OVF: zero out advantages for low-variance prompt groups
        if self._ovf_enabled and G >= 4:
            output["advantages"] = self._ovf_filter(output["advantages"], G)

        # TRLOO: scale advantages by N/(N-1) to correct self-inclusion bias
        if self._trloo_enabled:
            advantages = output["advantages"]
            if torch.isnan(advantages).any():
                output["advantages"] = self._trloo_scale_masked(advantages, G)
            else:
                scale = G / (G - 1.0)
                output["advantages"] = advantages * scale

        return output

    def _ovf_filter(self, advantages: torch.Tensor, G: int) -> torch.Tensor:
        """Optimal Variance Filtering: zero out low-variance prompt groups.

        Reshapes advantages into (num_prompts, G), computes per-group reward
        variance, and zeros out groups below the top_frac percentile. This
        focuses the GRPO update on informative samples where the model produced
        diverse outputs with different reward outcomes.
        """
        total = advantages.shape[0]
        if total % G != 0:
            return advantages  # Can't reshape cleanly, skip OVF

        num_prompts = total // G
        if num_prompts <= 1:
            return advantages  # Need multiple groups to filter

        grouped = advantages.view(num_prompts, G)
        # Compute variance per prompt group (use finite values only)
        finite_grouped = grouped.clone()
        finite_grouped[~torch.isfinite(finite_grouped)] = 0.0
        group_var = finite_grouped.var(dim=1)

        # Keep top_frac of groups by variance
        k = max(1, int(num_prompts * self._ovf_top_frac))
        _, top_indices = group_var.topk(k)

        # Create mask: zero out advantages for low-variance groups
        mask = torch.zeros(num_prompts, device=advantages.device, dtype=torch.bool)
        mask[top_indices] = True
        mask = mask.unsqueeze(1).expand_as(grouped).reshape(-1)

        filtered = advantages.clone()
        filtered[~mask] = 0.0

        kept_var = group_var[top_indices].mean().item()
        dropped_var = 0.0
        if num_prompts > k:
            all_indices = set(range(num_prompts))
            dropped_indices = list(all_indices - set(top_indices.tolist()))
            if dropped_indices:
                dropped_var = group_var[dropped_indices].mean().item()
        print(
            f"[ovf] kept {k}/{num_prompts} groups "
            f"(mean_var kept={kept_var:.4f}, dropped={dropped_var:.4f})"
        )
        return filtered

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
