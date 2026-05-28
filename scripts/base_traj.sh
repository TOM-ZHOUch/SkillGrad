# Stage 1 — Base-trajectory collection on the 200-task evolution pool.
#
# Runs the initial skill on every task to identify which ones the skill
# fails to solve. SkillGrad samples its training mini-batches from this
# failure pool. The grader writes per-task assessments under
#   results/base_trajectories/master_<M>_heldout_<H>/<MODEL>/
# plus a `failure_ids.json` that downstream stages read.
#
# Cost on the full 200-task sweep is roughly $3-5 per run on gpt-5.4.
#
# Usage:
#   bash scripts/base_traj.sh                # default: gpt-5.4
#   MODEL=gpt-4.1 bash scripts/base_traj.sh  # switch backbone

MODEL="${MODEL:-gpt-5.4}"
SKILLS_DIR="${SKILLS_DIR:-seeds}"

python -m runners.stream_runner base-trajectories \
    --model ${MODEL} \
    --master-seed 0 \
    --heldout-seed 42 \
    --skills-dir ${SKILLS_DIR} \
    --max-turns 20 \
    --executor-concurrency 4 \
    --grader-concurrency 1
