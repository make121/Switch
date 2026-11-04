#!/usr/bin/env bash
# Usage: bash scripts/eval_real_fall_recovery.sh [Hydra overrides...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_ckpt=${SWITCH_TASK_CHECKPOINT:-logs/MotionTracking/20260923_101737-switch3_recovery5_contact_refined_pilot-motion_tracking-g1_23dof_lock_wrist/model_241000.pt}
getup_ckpt=${SWITCH_RECOVERY_CHECKPOINT:-logs/MotionTracking/20261002_113221-recovery5_perturbed_rsi_stage3-motion_tracking-g1_23dof_lock_wrist/model_115000.pt}
if [[ ! -f "$getup_ckpt" ]]; then
    echo "Recovery checkpoint not found: $getup_ckpt (set SWITCH_RECOVERY_CHECKPOINT)." >&2
    exit 2
fi
bash scripts/eval_recovery_fsm.sh "$task_ckpt" \
    +device=cuda:0 \
    ++eval_name=real_fall_dual_policy \
    +env.config.noise_to_initial_level=0.0 \
    +domain_rand.push_robots=false \
    ++algo.config.eval_callbacks.skill_scheduler.config.recovery_fsm.policy_checkpoint="$getup_ckpt" \
    "$@"
