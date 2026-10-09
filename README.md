# SUGAR

Socially-grounded, Uncertainty-Gated, Augmented Recommender.

Two-stage next-item recommendation for social media: a frozen-LLM offline
stage (item refinement M1, profile building/editing M2, drift detection M3)
maintains temporally-consistent semantic user profiles; a lightweight
trainable online stage (social gate M4, SASRec-style backbone M5,
uncertainty-gated fusion M6) fuses the live sequence representation with
the projected profile, gated by a reliability score.

## Setup

pip install -e .
cp .env.example .env  # optional: DATA_DIR, OUTPUT_DIR, HF_TOKEN

## Pipeline

# 0) tests first -- NEVER start large runs with red tests
pytest tests/ -v

# 1) data (raw files must be in data/raw/ -- see data/prepare.py docstring)
python scripts/prepare_data.py --config configs/datasets/beauty.yaml
python scripts/prepare_data.py --config configs/datasets/yelp_filtered.yaml

# 2) offline stage (M1 cached; M2/M3 -> snapshot store)
python scripts/run_offline.py --phase all --dataset beauty   # or yelp_filtered

# 3) training (SUGAR + baselines, identical protocol)
python scripts/train.py --model sugar --dataset beauty --tag sugar_beauty
python scripts/train_baseline.py --model sasrec --dataset beauty --tag sasrec_beauty

# 4) everything else
make e3 e4 e5 e6 e7 e8 e9 ablation report
# or the full chain:
bash run_all.sh all

## Experiments

| Script | Experiment |
|---|---|
| `run_ablation.py` | E2: A1 (no profile), A2 (no drift), A3 (fixed gate), A4 (no social) |
| `inject_drift.py` | E3: synthetic drift injection + detector metrics |
| `corrupt_profiles.py` | E4.1: profile poisoning, SUGAR vs A3 |
| `vary_lag.py` | E4.2: staleness sweep over commit lag |
| `swap_graph.py` | E5: real / random / co-interaction graphs |
| `schema_stats.py` | E6: schema compliance + hallucination audit |
| `sensitivity.py` | E7: k and lambda grids |
| `cost_profile.py` | E8: token/call counts + online latency |
| `case_study.py` | E9: profile before/after + sim/rho series |

## Outputs

- `outputs/cache/` : LLM descriptors + encoder embeddings (M1, cached forever)
- `outputs/snapshots/` : snapshot stores (one pkl per dataset; `_nodrift` = A2)
- `outputs/results/` : one json per run (input to the table/plot generators)
- `outputs/report/` : LaTeX tables + PDF figures for the paper

## Reproducibility

Single seed (42) by default; `sugar/utils/seed.py` makes runs deterministic
up to GPU nondeterminism. All LLM calls are cached and greedy-decoded.
The leakage test (`tests/test_leakage.py`) enforces the temporal-consistency
contract of the snapshot protocol; run it before any large experiment.