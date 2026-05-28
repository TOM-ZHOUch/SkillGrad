# Stage 3 — Evaluate a trained skill on the held-out test split.
#
# Reads --skill-dir, runs the executor on every test task, and writes
# per-task trajectories plus a `summary.json` with the headline hard
# accuracy. The held-out split is determined by --master-seed and
# --heldout-seed; keep these consistent with the values used in train.sh.
#
# Usage:
#   bash scripts/eval.sh                          # default backbone: gpt-5.4
#   MODEL=gpt-4.1 bash scripts/eval.sh

MODEL="${MODEL:-gpt-5.4}"
RUN_ID="skillgrad_${MODEL}"

python -m runners.stream_runner eval \
    --skill-dir results/runs/${RUN_ID}/train/final_skill \
    --output-dir results/runs/${RUN_ID}/eval \
    --model ${MODEL} \
    --master-seed 0 --heldout-seed 42 \
    --executor-concurrency 4 --grader-concurrency 1
