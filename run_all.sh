#!/usr/bin/env bash
# Full 96h pipeline: offline -> training -> ablation -> E3/E4 -> E5/E7/E8/E9.
# Every stage is resumable: cached artifacts are skipped automatically.
# Usage: bash run_all.sh [beauty|yelp_filtered|all]
set -e

DATASET=${1:-all}

run_dataset () {
  local D=$1
  echo "=== [$D] offline M1 + M2/M3 ==="
  python scripts/run_offline.py --phase all --dataset $D

  echo "=== [$D] baselines ==="
  for M in gru4rec sasrec bert4rec unisrec; do
    python scripts/train_baseline.py --model $M --dataset $D \
        --tag ${M}_${D}
  done

  echo "=== [$D] SUGAR (A0) ==="
  python scripts/train.py --model sugar --dataset $D --tag sugar_${D}

  echo "=== [$D] ablations E2 ==="
  python scripts/run_ablation.py --dataset $D --variants a1 a2 a3 a4
}

if [ "$DATASET" = "beauty" ] || [ "$DATASET" = "all" ]; then
  run_dataset beauty
fi

if [ "$DATASET" = "yelp_filtered" ] || [ "$DATASET" = "all" ]; then
  run_dataset yelp_filtered
fi

echo "=== E4.1 poisoning curves (Beauty) ==="
for P in 0.2 0.4 0.6 0.8; do
  python scripts/corrupt_profiles.py --dataset beauty --p $P \
      --eval_model sugar --ckpt_tag sugar_beauty
  python scripts/corrupt_profiles.py --dataset beauty --p $P \
      --eval_model a3_fixed_gate --ckpt_tag a3_a3_fixed_gate_beauty
done

echo "=== E4.2 lag sweep ==="
python scripts/vary_lag.py --dataset beauty --ckpt_tag sugar_beauty

echo "=== E3 drift injection + offline rerun ==="
python scripts/inject_drift.py --dataset beauty

python scripts/run_offline.py --phase m2m3 --dataset beauty_drift

echo "=== E5 social graph substitution ==="
python scripts/swap_graph.py --dataset yelp_filtered \
    --ckpt_tag sugar_yelp_filtered

echo "=== E6 schema stats ==="
python scripts/schema_stats.py --dataset beauty --judge 200

echo "=== E7 sensitivity ==="
python scripts/sensitivity.py --dataset beauty \
    --param offline.window_size_k --values 5 10 20
python scripts/sensitivity.py --dataset beauty \
    --param model.lambda_align --values 0 0.01 0.1

echo "=== E8 cost profile ==="
python scripts/cost_profile.py --dataset beauty

echo "=== E9 case study ==="
python scripts/case_study.py --dataset beauty --ckpt_tag sugar_beauty

echo "=== tables and plots ==="
python analysis/make_tables.py
python analysis/make_plots.py

echo "ALL DONE. Results in outputs/results/, tables/plots in outputs/report/"