#!/usr/bin/env bash
# Usage: bash scripts/eval_recovery_fsm.sh CHECKPOINT [Hydra overrides...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash scripts/eval_recovery_fsm.sh CHECKPOINT [Hydra overrides...]' >&2
    exit 2
fi
recovery_ckpt=$1
shift
if [[ ! -f "$recovery_ckpt" ]]; then
    echo "Task checkpoint not found: $recovery_ckpt" >&2
    exit 2
fi
checkpoint_dir=$(dirname "$recovery_ckpt")
if [[ ! -f "$checkpoint_dir/config.yaml" && ! -f "$checkpoint_dir/../config.yaml" ]]; then
    echo 'Keep the training config.yaml beside the checkpoint (or its parent directory).' >&2
    exit 2
fi
recovery_motions=(
    Switch_data/skills/retargeted/141_12_poses_retarget.pkl
    Switch_data/skills/retargeted/141_14_poses_retarget.pkl
    Switch_data/skills/retargeted/141_19_poses_retarget.pkl
    Switch_data/recovery/retargeted_contact_refined/140_01_poses_retarget.pkl
    Switch_data/recovery/retargeted_contact_refined/140_02_poses_retarget.pkl
    Switch_data/recovery/retargeted_contact_refined/140_04_poses_retarget.pkl
    Switch_data/recovery/retargeted_contact_refined/140_08_poses_retarget.pkl
    Switch_data/recovery/retargeted_contact_refined/140_09_poses_retarget.pkl
)
for recovery_file in "${recovery_motions[@]}"; do
    if [[ ! -f "$recovery_file" ]]; then
        echo "Required Switch motion not found: $recovery_file" >&2
        exit 2
    fi
done
printf -v recovery_list '%s,' "${recovery_motions[@]}"
HYDRA_FULL_ERROR=1 LOGURU_LEVEL=INFO python humanoidverse/eval_agent.py \
    +recovery=eval_fsm \
    +checkpoint="$recovery_ckpt" +headless=false +eval_steps=6000 \
    +export_onnx=false \
    +robot.motion.motion_file="${recovery_motions[0]}" \
    +algo.config.eval_callbacks.skill_scheduler._target_=humanoidverse.agents.callbacks.skill_scheduler_callback.SkillSchedulerEvalCallback \
    +algo.config.eval_callbacks.skill_scheduler.config.skill_graph_path=Switch_data/unified_graph_contact_refined/skill_graph.json \
    +algo.config.eval_callbacks.skill_scheduler.config.scheduler_config_path=Switch_data/skill_graph/scheduler_config.yaml \
    "+algo.config.eval_callbacks.skill_scheduler.config.skill_pkl_files=[${recovery_list%,}]" \
    +algo.config.eval_callbacks.skill_scheduler.config.initial_skill=0 \
    +algo.config.eval_callbacks.transition_metrics._target_=humanoidverse.agents.callbacks.transition_metrics_callback.TransitionMetricsEvalCallback \
    '+algo.config.eval_callbacks.transition_metrics.config.output_path=${eval_log_dir}/recovery_metrics.json' \
    "$@"
