# Stage 2 — Train SkillGrad on the failure pool from stage 1.
#
# Each iteration samples a mini-batch of failed tasks, executes the current
# skill, runs the failure / contrastive diagnosers, updates the momentum
# state, and patches the skill in place. The final skill is written to
#   results/runs/<run_id>/train/final_skill/xlsx/
#
# Per-run cost on the default config (40 tasks, batch 4, 10 iterations) is
# roughly $6-15 on gpt-5.4.
#
# Usage:
#   bash scripts/train.sh                  # default backbone: gpt-5.4
#   MODEL=gpt-4.1 bash scripts/train.sh
#
# Override seeds or hyperparameters by editing the python call below, or by
# passing extra flags after the script's own arguments.

MODEL="${MODEL:-gpt-5.4}"
SKILLS_DIR="${SKILLS_DIR:-seeds}"

python -m pipeline.training \
    --data-dir data/benchmarks/spreadsheetbench \
    --skills-dir ${SKILLS_DIR} \
    --results-root results \
    --method skillgrad \
    --model ${MODEL} \
    --master-seed 0 --heldout-seed 42 --training-seed 0 \
    --n-train 40 --batch-size 4 --max-turns 30 --concurrency 4
