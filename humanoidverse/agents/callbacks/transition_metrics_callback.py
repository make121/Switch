"""Finite-run metrics for pre-sampled versus online transition evaluation."""

import json
from pathlib import Path

import numpy as np

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback


class TransitionMetricsEvalCallback(RL_EvalCallback):
    """Record tracking errors, termination causes and scheduler injections.

    The callback is read-only with respect to the environment and policy.  Use
    it together with ``eval_steps`` so ``on_post_evaluate_policy`` is reached.
    """

    def __init__(self, config, training_loop):
        super().__init__(config, training_loop)
        self.env = training_loop.env
        self.rows = []
        self.termination_counts = {}

    @staticmethod
    def _scalar(value, default=np.nan):
        if value is None:
            return float(default)
        try:
            return float(value[0].detach().cpu())
        except (IndexError, AttributeError, TypeError):
            return float(default)

    def on_post_eval_env_step(self, actor_state):
        env = self.env
        body_error = getattr(env, "dif_global_body_pos", None)
        if body_error is not None:
            per_body = body_error[0].norm(dim=-1)
            body_mean = float(per_body.mean().detach().cpu())
            body_max = float(per_body.max().detach().cpu())
        else:
            body_mean = body_max = float("nan")

        dof_error = None
        if hasattr(env, "ref_joint_pos"):
            dof_error = (env.simulator.dof_pos[0] - env.ref_joint_pos[0]).abs()

        term_names = []
        for name, value in getattr(env, "reset_buf_terminate_by", {}).items():
            if bool(value[0]):
                term_names.append(name)
                self.termination_counts[name] = self.termination_counts.get(name, 0) + 1

        self.rows.append({
            "step": int(actor_state.get("step", len(self.rows))),
            "body_error_mean_m": body_mean,
            "body_error_max_m": body_max,
            "dof_error_l1_rad": float(dof_error.sum().detach().cpu())
            if dof_error is not None else float("nan"),
            "done": bool(actor_state["dones"][0]),
            "termination": term_names,
            "is_buffer": bool(getattr(env, "ref_is_buffer", [[False]])[0][0]),
            "kappa": self._scalar(getattr(env, "ref_kappa", None), 0.0),
        })
        return actor_state

    def on_post_evaluate_policy(self):
        scheduler_history = []
        for callback in self.training_loop.eval_callbacks:
            if hasattr(callback, "injection_history"):
                scheduler_history = list(callback.injection_history)
                break

        def values(key):
            return np.asarray([r[key] for r in self.rows], dtype=np.float64)

        body_mean = values("body_error_mean_m")
        body_max = values("body_error_max_m")
        dof_l1 = values("dof_error_l1_rad")
        buffer_rows = [r for r in self.rows if r["is_buffer"]]
        summary = {
            "num_steps": len(self.rows),
            "done_steps": sum(r["done"] for r in self.rows),
            "termination_counts": self.termination_counts,
            "body_error_mean_m": float(np.nanmean(body_mean)),
            "body_error_p95_m": float(np.nanpercentile(body_mean, 95)),
            "body_error_max_m": float(np.nanmax(body_max)),
            "dof_error_l1_mean_rad": float(np.nanmean(dof_l1)),
            "buffer_steps": len(buffer_rows),
            "buffer_body_error_mean_m": float(np.mean(
                [r["body_error_mean_m"] for r in buffer_rows]
            )) if buffer_rows else None,
            "scheduler_injections": scheduler_history,
        }
        output = Path(str(self.config.output_path))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump({"summary": summary, "steps": self.rows}, f, indent=2)
        print(f"Transition metrics written to {output}")
