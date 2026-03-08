# KernelForge Documentation

## Active Documents

| Document | Role |
|----------|------|
| [KERNELFORGE_FINAL_PRD.md](KERNELFORGE_FINAL_PRD.md) | Main source of truth for architecture, rollout order, and launch criteria |
| [GRPO_DEEP_DIVE.md](GRPO_DEEP_DIVE.md) | GRPO/TRLOO strategy, rollout design, and hackathon training posture |
| [OPENENV_AUDIT_PLAN_3.md](OPENENV_AUDIT_PLAN_3.md) | OpenEnv contract review and architecture audit |
| [CLAUDE.md](CLAUDE.md) | Navigation hub and quick-reference for the current repo |

## Skill Documents

| File | Role |
|------|------|
| [skills/CUDA_AGENT.md](skills/CUDA_AGENT.md) | CUDA-Agent task/environment prior |
| [skills/DOUBLEGRAPH_A100.md](skills/DOUBLEGRAPH_A100.md) | DoubleGraph A100 engineering prior |
| [skills/SKYDISCOVER_ADAEVOLVE_EVOX.md](skills/SKYDISCOVER_ADAEVOLVE_EVOX.md) | Search-system prior |
| [skills/KERNELGYM_DR_KERNEL.md](skills/KERNELGYM_DR_KERNEL.md) | KernelGYM backend architecture and Dr. Kernel/TRLOO framing |
| [skills/CLAUDE.md](skills/CLAUDE.md) | Skill index |

## Runtime Posture

The current documented runtime posture is:

- custom lightweight OpenEnv wrapper at the edge
- KernelForge task/reward/rollout logic in the middle
- CoreWeave/Northflank eval service as the primary backend
- Modal preserved as fallback only

External references:
- [OpenEnv overview](https://meta-pytorch.github.io/OpenEnv/)
- [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)
- [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM)
- [Dr. Kernel paper](https://arxiv.org/abs/2602.05885)

## Archive

Historical docs and older research material remain under [archive/](archive/).
