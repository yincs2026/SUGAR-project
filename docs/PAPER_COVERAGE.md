# Manuscript-to-code coverage

This inventory distinguishes executable coverage from exact recovery of unspecified original choices. It is not a blanket assurance that every empirical claim is true. Outputs below are produced by execution; paper reference scores are never copied into experiment results. The default research suite requires real backends. CI and unit tests use separate fixtures.

| Manuscript method or experiment | Executable implementation | Configuration / measured output |
|---|---|---|
| M1, Eq. 5: frozen JSON item refinement, cached 768-dimensional BGE | `semantics.py`, `offline.py` | `semantic_backend: real`; descriptors, embeddings, usage, integrity manifest |
| M2, Eqs. 6–9: initial profile, weighted topics, disjoint consolidation and revision | `semantics.py`, `offline.py` | `profile_embedding: weighted`, checkpoint 20, edit 10; event evidence, profiles, snapshots |
| M3, Eqs. 10–11: cosine drift, personalized/global threshold, two strikes, lag/cooldown | `offline.Detector` | xi 1, lag 10, cooldown 50; detector trace and committed snapshots |
| M4, Eqs. 12–15: top-L affinity, positive learned temperature, social gate | `features.py`, `model.py` | L=10; no-neighbor gamma=0; per-user gamma |
| M5, Eqs. 16–20: causal SASRec, shared projection, freshness/coverage reliability, vector fusion | `model.py` | dim 64, two blocks/heads, maxlen 200, kappa 50; per-user rho/gates |
| M6, Eqs. 21–23: BCE and uniform negatives, shared-projection InfoNCE | `model.py`, `train.py` | alignment .05, temperature .1; losses and trainable checkpoints |
| Eqs. 24–25: full-catalog HR@5/10/20, NDCG@10 | `train.py` | exact streamed catalog ranks, per-user outputs |
| Timestamped train-only personal/social snapshot joins | `offline.py`, `features.py` | default leak audit; integrity hashes; future evidence rejection tests |
| Table 4: three datasets, iterative pre-split 5-core, chronological holdouts | `data.py`, conversion scripts | `suite-paper.yaml`; dataset statistics; `tables/table_4.csv` |
| Table 5: SUGAR, five seeds, mean/std | `suite.py`, `experiments.aggregate` | `models/sugar/seed-*`; `tables/table_5.csv` |
| Table 6: sparsity buckets, HR and social gate | `train.evaluate`, `experiments.aggregate` | training-length buckets including explicit below5; `tables/table_6.csv` |
| Table 7: no drift/social/alignment, fixed fusion, snapshot-discipline diagnostic | `suite.VARIANTS`, `factory.py` | independent training/artifacts, validation-selected fixed fusion; `tables/table_7.csv` |
| Table 8: clean/random donor/stale profiles, SUGAR/fixed/SASRec, gate and forced-open error correlation | `features.py`, `experiments.stress_test`, `suite.py` | deterministic random temporally visible donors; `stress_comparison.json`; `tables/table_8.csv` |
| Table 9: GRU4Rec, Caser, SASRec, BERT4Rec, LM-Feature, RLMRec, LLaRA, LightGCN+, SocialMF | `baselines.py`, `llara.py`, `factory.py`, `suite.py` | all local trainable paths; 20 validation tuning trials; `tables/table_9.csv` |
| Independent static baseline profiles; RLMRec contrastive/generative variants | `coherence.py`, `baselines.py` | static profile cache; `rlm_variant: con` or `gen` |
| LLaRA pretrained LM, frozen sequential item vectors, LoRA, curriculum/hybrid prompts | `llara.py`, `suite.py` | immutable real LM revision, pretrained SASRec dependency, adapter checkpoints |
| Weighted profile vs whole JSON; recency-weighted window | `semantics.py`, `features.py`, `suite.py` | `json-profile`, `recency-window` variants |
| Shared vs independent alignment projection | `model.py`, `suite.py` | `separate-projection` variant |
| Full vs sampled neighborhood | `features.py`, `suite.py` | `sampled-neighbors` variant |
| BCE vs BPR | `model.py`, `suite.py` | `bpr` variant |
| Single vs two strikes; fixed profile; committed vs live freshness score | `offline.py`, `features.py`, `suite.py` | `single-strike`, `no-drift`, `live-score` |
| Eqs. 26–28: independent metadata JS labels, precision/recall/delay/calls | `experiments.drift_analysis` | pooled across all three datasets; 90th percentile; tolerance 3 checkpoints |
| JS label robustness at 85th and 95th percentiles | `coverage.drift_percentiles` | `drift_percentile_robustness.json`, per-percentile reports |
| Global tau .55–.85; personalized xi .5/1/1.5/2 | `suite.py`, `coverage.sweep_frontier` | sweeps, single seed per point, measured Pareto front and 40-call budget |
| Lag 0/5/10/20, L 5/10/20/50, kappa 10/25/50/100 | `suite.py` | regenerated policy artifacts and trained sweep checkpoints |
| Train-only shared-item >=3 proxy graph for SUGAR/LightGCN+ | `data.proxy_graph`, `suite.py` | proxy protocol audits, models and summaries |
| Frozen encoder quality ARI | `coherence.encoder_quality` | metadata category clustering; `encoder_quality.json` |
| Paired seed-level significance | `experiments.paired_test`, `suite.py` | paired validation-compatible seeded comparisons |
| Model and end-to-end latency p50/p95, throughput, memory | `experiments.benchmark`, `benchmark_pipeline` | batch-one synced measurements; separate timing scopes |
| Module calls, token/time shares, preprocessing time, fallback rate | `coverage.offline_costs`, `offline.py` | `offline_costs.json`, `offline_timing.json`, actual/cache token separation |
| Parameter overhead | `coverage.parameter_counts` | backbone vs augmentation, frozen vs trainable; `parameter_counts.json` |
| Amortized semantic projection | `model.compute_semantic_branch` | exact-input inference cache with training/device/weight invalidation |
| Reported failed Epinions LoRA seeds | `coverage.seed_outcomes`, suite failure records | completed/failed/incomplete status; unpublished convergence criterion is not invented |
| Compare published numeric results | `verification.py` | Table 4 and Table 5/9 reference comparison only; not training inputs |

## Remaining protocol decisions

* **Original real data are not included.** The manuscript's Yelp review-item definition is incompatible with ordinary one-review/one-interaction 5-core preprocessing. The business-item converter is an explicit alternative, not the unidentified original construction. Douban/Epinions exact releases, subsets, text and category joins must be supplied. Missing timestamps/categories are not fabricated.
* **Baseline adaptation:** local baseline paths exist, but LLaRA full-catalog conditional likelihood is an explicit adaptation; the manuscript's unidentified router cannot be recovered. See `BASELINES.md` for every baseline.
* **Fallback:** invalid-item recovery extracts hashtags and named entities using a real version-checked spaCy checkpoint. The manuscript does not identify its original entity model; the configured checkpoint is an explicit choice. Profile failures abort.
* **Snapshot amortization:** a live recent vector and evolving visible neighbors change the semantic input between requests. Exact-input caching is implemented; the undocumented claim of one constant projection per entire chunk requires the `committed` recent-vector choice and stable neighbors.
* Original prompts, model revisions, negative pool size, sampled-neighbor budget, convergence rule, and several other details are not supplied. Executable choices are itemized in `REPRODUCTION_AUDIT.md`.

These limitations prevent certifying literal implementation of **every** manuscript claim. They do not get resolved by calling the README an official repository. The listed experiments have execution paths, but real benchmark completion and original protocol identity require the missing inputs and decisions.
