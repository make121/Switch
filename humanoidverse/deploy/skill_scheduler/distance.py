"""Normalized state distance for the online skill scheduler (spec 1.3).

This is the SINGLE implementation of the state distance / similarity used by:
  (a) deploy-time cross-skill edge weights (d_uv),
  (b) entry check,
  (c) safety thresholds (A / B),
  (d) offline calibration (calibrate.py).

Do not fork it per call site. Semantics: smaller = closer (it is a distance,
NOT a [0, 1] bounded similarity). `sim <= A` means close enough to attach
directly; `sim >= B` means too far, trigger e-stop.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NodeState:
    """State features of a graph node or of the live robot.

    q:     joint positions (23,)
    q_dot: joint velocities (23,)
    p_hat: root translation (3,), local frame (global x-y / yaw removed at
           graph construction time; live robot state must be transformed the
           same way before calling sim()).
    """

    q: np.ndarray
    q_dot: np.ndarray
    p_hat: np.ndarray


def component_l1(a: np.ndarray, b: np.ndarray) -> float:
    """L1 norm of the difference between two feature vectors."""
    return float(np.abs(np.asarray(a, dtype=np.float64)
                        - np.asarray(b, dtype=np.float64)).sum())


def sim(x: NodeState, node: NodeState,
        sigma_q: float, sigma_qdot: float, sigma_p: float,
        w_q: float = 1.0, w_qdot: float = 1.0, w_p: float = 1.0) -> float:
    """Component-normalized weighted distance (spec 1.3, confirmed option 2).

    d = w_q * ||x.q - node.q||_1 / sigma_q
      + w_qdot * ||x.q_dot - node.q_dot||_1 / sigma_qdot
      + w_p * ||x.p_hat - node.p_hat||_1 / sigma_p

    sigma_* are offline statistics (std of the per-component L1 distance over
    all graph edges), computed once by calibrate.py and stored in config.
    """
    d_q = component_l1(x.q, node.q) / sigma_q
    d_qdot = component_l1(x.q_dot, node.q_dot) / sigma_qdot
    d_p = component_l1(x.p_hat, node.p_hat) / sigma_p
    return w_q * d_q + w_qdot * d_qdot + w_p * d_p
