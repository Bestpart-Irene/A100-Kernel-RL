"""
KernelForge OpenEnv Environment for target-GPU CUDA kernel RL training.

OpenEnv is an independent framework from Meta-PyTorch (NOT a Gymnasium
extension). HTTP client-server architecture with Docker container isolation.
Correct install (uv): uv add "openenv-core[core]>=0.2.1" (NOT "openenv")
"""
import os
from typing import Any
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State
from openenv_env.models import KernelForgeAction, KernelForgeObservation
from openenv_env.gpu_registry import get_gpu_spec
from openenv_env.reward import compute_reward
from openenv_env.skill_builder import build_skill_md
from openenv_env.task_pool import TaskPool
from training.task_support import (
    build_modal_payload,
    normalize_eval_result,
    normalize_task_row,
    task_interface_contract,
)


class KernelForgeEnv(Environment):
    """
    RL environment: agent submits CUDA source -> target GPU compiles/verifies/benchmarks.

    Action space: CUDA source code string
    Observation: SKILL.md + task + reference code + feedback + history
    Reward: discrete milestones {-1, +1, +2, +3}
    Max turns: 3 (matching Dr. Kernel; configurable via KERNELFORGE_MAX_TURNS)
    Context: 128K tokens
    """

    def __init__(self, task_pool: TaskPool | None = None):
        super().__init__()
        self.target_gpu = os.getenv("KERNELFORGE_TARGET_GPU", "a100").lower()
        self.gpu_spec = get_gpu_spec(self.target_gpu)
        self.task_pool = task_pool or TaskPool.load()
        self.history = []
        self.turn = 0
        self.max_turns = int(os.getenv("KERNELFORGE_MAX_TURNS", "3"))
        self.best_reward = -1.0
        self.best_code = None
        self.original_baseline_ms = None
        self.doublegraph_baseline_ms = None
        self.current_task: dict[str, Any] | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> KernelForgeObservation:
        """Reset environment with a task sampled from the pool.

        Args:
            seed: Random seed for reproducible task sampling.
            episode_id: Optional episode identifier.
            task_id: If provided, use this specific task instead of sampling.
        """
        self.history = []
        self.turn = 0
        self.best_reward = -1.0
        self.best_code = None
        self.original_baseline_ms = None
        self.doublegraph_baseline_ms = None
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        # Sample task from pool
        task_row = self.task_pool.sample(task_id=task_id, seed=seed)
        self.current_task = normalize_task_row(task_row)

        # Load cached baselines if available
        tid = self.current_task.get("task_id", "")
        cached = self.task_pool.get_cached_baselines(tid) if tid else None
        if cached:
            self.original_baseline_ms = cached.get("eager_ms")
            self.doublegraph_baseline_ms = cached.get("compile_ms")

        # Profile WCC baselines on first call (graph tasks only)
        if (
            self.original_baseline_ms is None
            and self.current_task.get("evaluation_backend") == "wcc"
        ):
            try:
                baselines = self._dispatch("profile_baselines")
                self.original_baseline_ms = baselines.get("original_ms")
                self.doublegraph_baseline_ms = baselines.get("doublegraph_ms")
            except Exception:
                pass

        # Build initial observation with SKILL.md + task + reference code + contract
        task_prompt = self.current_task.get("prompt", "")
        contract = task_interface_contract(self.current_task)
        task_code = self.current_task.get("task_code", "")

        obs_parts = [build_skill_md(self.target_gpu)]
        obs_parts.append(f"\n\n---\n\nTask: {task_prompt}")
        if task_code:
            obs_parts.append(
                f"\n\nReference implementation:\n```python\n{task_code}\n```"
            )
        obs_parts.append(f"\n\n{contract}")

        return KernelForgeObservation(
            text="".join(obs_parts),
            baseline_original_ms=self.original_baseline_ms,
            baseline_doublegraph_ms=self.doublegraph_baseline_ms,
            hardware=self.gpu_spec,
            reward=0.0,
            done=False,
            turn=self.turn,
            best_reward=self.best_reward,
            info={
                "phase": "reset",
                "task_id": tid,
                "evaluation_backend": self.current_task.get("evaluation_backend"),
                "ops": self.current_task.get("ops", []),
                "pool_size": self.task_pool.size,
            },
            graph_properties=self.current_task.get("graph_properties"),
            topology_type=self.current_task.get("topology"),
        )

    def step(
        self,
        action: KernelForgeAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> KernelForgeObservation:
        """Execute one environment step from CUDA source code action."""
        self.turn += 1
        self._state.step_count = self.turn

        try:
            fn_name, payload = build_modal_payload(
                action.cuda_code,
                self.current_task,
                baseline_orig_ms=self.original_baseline_ms,
                baseline_dg_ms=self.doublegraph_baseline_ms,
            )
            result = normalize_eval_result(self._dispatch(fn_name, payload))
        except Exception as exc:
            result = {
                "compiles": False,
                "correct": False,
                "runtime_ms": 0.0,
                "runtime_stats": {},
                "speedup_vs_orig": 0.0,
                "speedup_vs_dg": 0.0,
                "error": str(exc),
            }

        # Compute speedups and reward via canonical function
        if not result.get("compiles"):
            reward = -1.0
            su_orig = 0
            obs = (f"COMPILATION FAILED (turn {self.turn}/{self.max_turns}):\n"
                   f"{result.get('error', 'Unknown error')[:1500]}")
        elif not result.get("correct"):
            reward = -1.0
            su_orig = 0
            obs = (f"VERIFICATION FAILED (turn {self.turn}/{self.max_turns}):\n"
                   f"{result.get('verifier_msg', result.get('error', 'Unknown failure'))}")
        else:
            rt = float(result["runtime_ms"])
            su_orig = (
                self.original_baseline_ms / rt
                if self.original_baseline_ms and rt > 0
                else float(result.get("speedup_vs_orig", 0.0) or 0.0)
            )
            su_dg = (
                self.doublegraph_baseline_ms / rt
                if self.doublegraph_baseline_ms and rt > 0
                else float(result.get("speedup_vs_dg", 0.0) or 0.0)
            )

            reward = compute_reward(
                compiled=True,
                correct=True,
                speedup_vs_eager=su_orig,
                speedup_vs_compile=su_dg,
            )

            # Cache baselines from ops6k eval result
            if result.get("baseline_eager_ms") and not self.original_baseline_ms:
                self.original_baseline_ms = result["baseline_eager_ms"]
            if result.get("baseline_compile_ms") and not self.doublegraph_baseline_ms:
                self.doublegraph_baseline_ms = result["baseline_compile_ms"]
            tid = (self.current_task or {}).get("task_id", "")
            if tid and (self.original_baseline_ms or self.doublegraph_baseline_ms):
                self.task_pool.cache_baselines(tid, {
                    "eager_ms": self.original_baseline_ms or 0.0,
                    "compile_ms": self.doublegraph_baseline_ms or 0.0,
                })

            backend = (self.current_task or {}).get("evaluation_backend", "ops6k")
            if backend == "ops6k":
                obs = (f"BENCHMARK (turn {self.turn}/{self.max_turns}):\n"
                       f"  Runtime: {rt:.3f}ms\n"
                       f"  vs eager: {su_orig:.2f}x\n"
                       f"  vs torch.compile: {su_dg:.2f}x")
            else:
                obs = (f"BENCHMARK (turn {self.turn}/{self.max_turns}):\n"
                       f"  Runtime: {rt:.3f}ms\n"
                       f"  vs cuGraph: {su_orig:.2f}x")
                if su_dg:
                    obs += f"\n  vs doubleGraph: {su_dg:.2f}x"
            obs += f"\n  Stats: {result.get('runtime_stats', {})}"

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_code = action.cuda_code

        done = (self.turn >= self.max_turns) or (reward >= 3.0)

        # Time-travel snapshots
        self.history.append({
            "turn": self.turn,
            "reward": reward,
            "obs_summary": obs[:200],
        })

        # Include history in observation for multi-turn context
        if len(self.history) > 1:
            history_ctx = "\n--- Previous attempts ---\n"
            for h in self.history[:-1]:
                history_ctx += (f"Turn {h['turn']}: reward={h['reward']}, "
                               f"{h['obs_summary'][:80]}\n")
            obs = history_ctx + "\n--- Current result ---\n" + obs

        return KernelForgeObservation(
            text=obs,
            reward=reward,
            done=done,
            turn=self.turn,
            best_reward=self.best_reward,
            info={
                "turn": self.turn,
                "best_reward": self.best_reward,
                "speedup": su_orig if result.get("correct") else 0,
                "evaluation_backend": (self.current_task or {}).get("evaluation_backend"),
                "task_id": (self.current_task or {}).get("task_id", ""),
            },
            graph_properties=(self.current_task or {}).get("graph_properties"),
            topology_type=(self.current_task or {}).get("topology"),
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self.turn,
            history=self.history,
            best_reward=self.best_reward,
        )

    def close(self):
        """Clean up episode resources."""
        self.history = []
        self.current_task = None

    def _dispatch(self, fn_name, payload=None):
        """Dispatch to configured eval backend (CoreWeave or Modal)."""
        from openenv_env.eval_backend import dispatch_eval
        return dispatch_eval(fn_name, payload)
