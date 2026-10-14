# SUGAR

**LLM-Augmented Personalized Sequential Recommendation in Social Media**

SUGAR combines language-model-derived user profiles, social information, and sequential recommendation through reliability-gated fusion.

The framework consists of two stages:

- **Offline:** item semantic refinement, user profile construction and consolidation, and drift-triggered profile revision.
- **Online:** temporal profile retrieval, social aggregation, SASRec sequence encoding, and reliability-gated representation fusion.

The semantic stage uses frozen **Qwen2.5-7B-Instruct** and **BGE-base-en-v1.5** models. The recommendation stage is trained with a recommendation loss and semantic alignment.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -e '.[semantic,analysis,llara,fallback]'

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

For CUDA 12.1, install PyTorch before installing the package:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

## Data

Place the normalized datasets in:

```text
data/raw/yelp/
data/raw/douban-book/
data/raw/epinions/
```

Each dataset directory contains:

| File | Format |
|---|---|
| `interactions.csv` | `user_id,item_id,timestamp[,rating]` |
| `items.jsonl` | One JSON object per line containing `item_id,text,category[,available_at]` |
| `social.csv` | `user_id,neighbor_id[,timestamp]` |

Preprocessing applies iterative 5-core filtering, chronological ordering, and leave-one-out validation and test splits. Douban-Book interactions with ratings below 4 are removed.

Dataset preparation and source conversion are described in [DATA.md](docs/DATA.md).

## Model Configuration

Resolve immutable pretrained-model revisions for each dataset:

```bash
sugar resolve-models \
  --config configs/paper.yaml \
  --out configs/resolved-yelp.yaml

sugar resolve-models \
  --config configs/douban-book.yaml \
  --out configs/resolved-douban-book.yaml

sugar resolve-models \
  --config configs/epinions.yaml \
  --out configs/resolved-epinions.yaml
```

## Offline Language Model Server

Run the language model server in a separate environment:

```bash
python3.11 -m venv .venv-llm
source .venv-llm/bin/activate

pip install -r requirements-llm.txt

python scripts/serve_llm.py \
  --config configs/resolved-yelp.yaml \
  --provenance artifacts/llm_server.json
```

Keep the server running. Execute the following commands from another terminal with `.venv` activated.

## Prepare Data and Semantic Profiles

The following example uses Yelp:

```bash
sugar prepare \
  --config configs/resolved-yelp.yaml \
  --raw data/raw/yelp \
  --out data/prepared/yelp

sugar offline \
  --config configs/resolved-yelp.yaml \
  --data data/prepared/yelp/data.json \
  --cache cache/yelp \
  --out artifacts/yelp

sugar audit \
  --config configs/resolved-yelp.yaml \
  --data data/prepared/yelp/data.json \
  --artifacts artifacts/yelp \
  --out artifacts/yelp/leakage_audit.json
```

Use the corresponding dataset configuration and paths for Douban-Book and Epinions.

## Training

Train SUGAR with one seed:

```bash
sugar train \
  --config configs/resolved-yelp.yaml \
  --data data/prepared/yelp/data.json \
  --artifacts artifacts/yelp \
  --out runs/yelp/seed-42 \
  --device cuda \
  --seed 42
```

Run all configured seeds:

```bash
sugar multi-seed \
  --config configs/resolved-yelp.yaml \
  --data data/prepared/yelp/data.json \
  --artifacts artifacts/yelp \
  --out runs/yelp/main \
  --device cuda
```

Model selection uses validation NDCG@10. Each run saves its configuration, provenance, checkpoints, and evaluation outputs.

## Evaluation

Evaluate the selected checkpoint over the full item catalog:

```bash
sugar evaluate \
  --data data/prepared/yelp/data.json \
  --artifacts artifacts/yelp \
  --run runs/yelp/seed-42 \
  --out runs/yelp/seed-42/evaluation.json \
  --device cuda
```

Reported ranking metrics are **HR@5**, **HR@10**, **HR@20**, and **NDCG@10**.

## Complete Experiment Suite

Execute the benchmark models and experiments across all three datasets:

```bash
sugar suite \
  --spec configs/suite-paper.yaml \
  --out runs/paper-suite \
  --device cuda
```

Resume an interrupted suite:

```bash
sugar suite \
  --spec configs/suite-paper.yaml \
  --out runs/paper-suite \
  --device cuda \
  --resume
```

The suite includes:

- SUGAR and nine comparison baselines.
- Five-seed evaluation and paired comparisons.
- Module ablations and fixed-fusion tuning.
- Clean, corrupted, and stale profile experiments.
- Drift detection and percentile robustness.
- Threshold, revision lag, neighborhood size, and reliability sensitivity.
- Profile representation and alignment variants.
- Proxy social graph experiments.
- Encoder clustering quality.
- Latency, throughput, parameter counts, and offline processing costs.

Baseline implementations and evaluation protocols are described in [BASELINES.md](docs/BASELINES.md).

## Results

Results are saved under `runs/paper-suite/`.

| Output | Contents |
|---|---|
| `computed_results.csv` | Aggregated ranking metrics |
| `tables/table_4.csv`–`table_9.csv` | Computed experiment tables |
| `drift_detection.json` | Drift detection measurements |
| `drift_percentile_robustness.json` | Drift label sensitivity |
| `<dataset>/models/` | Baseline and SUGAR runs |
| `<dataset>/variants/` | Ablation runs |
| `<dataset>/sweeps/` | Sensitivity experiments |
| `<dataset>/stress_comparison.json` | Profile stress experiments |
| `<dataset>/encoder_quality.json` | Encoder clustering measurements |
| `<dataset>/offline_costs.json` | Semantic processing costs |
| `<dataset>/parameter_counts.json` | Model parameter counts |
| `<dataset>/seed_outcomes.json` | Seed execution status |
| `suite_manifest.json` | Experiment execution manifest |

Export the computed tables separately:

```bash
sugar export-tables \
  --suite runs/paper-suite \
  --out runs/paper-suite/tables
```

## Tests

```bash
python -m unittest discover -s tests -v
```
