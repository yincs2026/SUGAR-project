# Reproduction audit

The supplied manuscript defines an architecture and reports results, but it is not a complete executable specification. This implementation follows its equations while explicitly documenting choices that cannot be recovered from the Word file. The reported benchmark scores remain unverified.

## Module mapping

| Paper specification | Implementation | Status |
|---|---|---|
| M1 structured refinement and Eq.5 | `semantics.py`, `offline.py` | JSON-guided vLLM API, strict schemas, BGE cached embeddings |
| M2 initial build, consolidation, revision Eq.6 | `offline.py` | Event-index evidence tracking; three distinct prompts |
| Recent mean and profile representation Eqs.7–9 | `offline.py`, `semantics.py` | Uniform recent average; normalized weighted topic encoding |
| Trigger and personalized threshold Eqs.10–11 | `Detector` | Past-only statistics, two strikes, revision commit and cooldown |
| Social aggregation and blending Eqs.12–15 | `features.py`, `model.py` | Top-L affinity selection, learned positive temperature, trainable gate |
| Backbone and projection Eqs.16–17 | `model.py` | Causal SASRec blocks; shared profile/item projection |
| Reliability and fusion Eqs.18–20 | `model.py` | Trainable unconstrained a/b/bias; vector sigmoid gate |
| BCE and InfoNCE Eqs.21–23 | `model.py`, `train.py` | Uniform train-only negative sampling; BPR option |
| Ranking and HR/NDCG Eqs.24–25 | `train.py` | Full-catalog chunked ranking |
| JS labels and matching Eqs.26–28 | `experiments.py` | Pooled independent metadata, tolerance of three checkpoints |
| Temporal user/neighbor snapshots | `offline.py`, `features.py` | Commit-position and absolute-time constraints; integrity hashes |

The model does not multiply the fusion gate by reliability or constrain a/b to stay positive: neither modification appears in the paper. Consequently graceful degradation is an empirical hypothesis, not a mathematical guarantee. Social neighbors are not additionally downweighted by their own reliability, consistent with the stated future-work limitation.

## Blocking inputs for exact numerical reproduction

1. Original raw dataset releases, exact subset selection, timestamp semantics and processed ID maps.
2. Immutable original Qwen/BGE revisions, all cached outputs, and exact descriptor/profile serialization.
3. Original seeds, optimizer details, negative-sampling implementation, sequence batching and initialization.
4. Exact profile build/edit/revision prompts and decoding/software/hardware provenance.
5. Baseline implementations, revisions, tuning configurations, selected checkpoints and seed-level results.
6. Original evaluation candidate masks, treatment of repeat items, and drift-category mapping.

The seed list 42–46 is an implementation choice. Resolving today's `main` model revision makes a new run reproducible; it does not recover the paper's old revision. Cache keys and manifests prevent mixing new outputs with old ones.

## Dataset inconsistencies

**Yelp item definition.** Section 4.2 calls business reviews the items and mentions title + categories + the first 128 review words. Under the usual one-review/one-author-interaction mapping, every review item has degree one; a 5-core cannot retain the reported catalog. Business IDs instead have many interactions, but using an interaction's own review text as a static business descriptor would expose target behavior/content. The converter supplies a clearly marked alternative using business names/categories only. Exact reproduction needs the author's precise item/text construction and availability rules.

**Douban/Epinions provenance.** There are multiple public variants, and some lack trustworthy timestamps, product/book text or category metadata. This repository does not invent missing values, scrape replacement texts or assert a download corresponds to the reported Table 4 subset.

**Core versus sparsity buckets.** A pre-split minimum of five interactions yields as few as three training interactions after two holdouts. Table 6's smallest bucket starts at five training interactions. This implementation exposes `below5`; exact percentages require the original treatment of those users.

**Profile coverage.** With m_chk=20 and lag=10, the first profile is visible only after 30 observed training interactions, despite n0=5. Sparse users therefore have a learnable default profile, possibly augmented by neighbors. This follows the checkpoint specification rather than silently building profiles at interaction five.

## Explicit implementation choices

| Unspecified or conflicting detail | Choice here | Consequence |
|---|---|---|
| Topic clustering | Exact normalized topic-string groups, counts, styles, up to three summaries | No unreported clustering model is assumed |
| Profile prompts/schema | Concrete strings in `semantics.py` | Original prompts needed to match original caches |
| Excluded topics from positive-only data | Remain empty unless actual avoidance evidence is supplied | Absence of engagement does not establish dislike |
| Initial reliability coefficients | a=b=1, bias=0 | Positive initialization only; subsequently unconstrained |
| Social temperature initialization | 0.1, softplus parameterization | Learnable strictly positive beta |
| Personalized warmup | Global tau=0.7 for first two past scores; population std | Not specified by manuscript |
| Consolidation accounting | Distinct event positions, not unique item IDs | Repeated events contribute once each |
| Edit/revision overlap | Consolidation first, then revision; both calls counted | Matches Algorithm 1 order |
| Revision timing | Staged artifact; no overlapping edit while a job is pending | Resolves text/Algorithm 1 asynchrony conflict |
| Cooldown origin | Commit position plus 50 new interactions | Follows prose rather than immediate Algorithm 1 cooldown |
| Lag past final training event | Snapshot never visible | Does not use a holdout to determine commit |
| Neighbor availability | Strict commit timestamp < target timestamp | Avoids ambiguous simultaneous-event leakage |
| Dynamic social signal | Recent vector and visible neighbors recomputed per query | Prose's piecewise-constant M4 conflicts with live R and neighbor commits |
| Test behavioral context | M5 sees validation event; M2–M4 use training events only | Standard leave-one-out sequence context; heldout semantic exclusion retained |
| Recommendation negatives | Uniform with replacement outside the user's full TRAIN sequence | No validation/test negative exclusions |
| Alignment negative pool | 64 samples with replacement outside TRAIN sequence | Number not specified; configurable |
| Loss reduction | Per-target mean, negative sum | Preserves relative BCE/InfoNCE scaling per position |
| Batch construction | Each next-item prefix is a training example | Covers every position; differs from unspecified batched-user sampling |
| Sequence limit | Most recent 200 prefix items, right padded | Computationally more costly than processing all targets in one user pass |
| Backbone internals | SASRec-style pre-normalized attention and position-wise feedforward | Original code/LayerNorm placement needed for exact identity |
| Item initialization | Normal std=0.02; padding zero | Original initialization not supplied |
| Seen candidates | Mask seen history but restore repeat heldout target | Configurable, changes reported ranks |
| Candidate tie break | Ascending mapped item ID | Stable deterministic comparison |
| Catalog availability | Items available strictly before request | Input availability dates should be source-grounded |
| Corruption | Uniform random different-user donor among temporally visible profiles, with a fixed corruption seed | Corruption sampling seed not supplied |
| Staleness | Latest committed snapshot built before half the training history | Falls back to earliest visible snapshot if none eligible |
| Corruption reliability | Keep substituted snapshot metadata; no hidden oracle corruption flag | Whether paper recomputed freshness is unspecified |
| Spearman error | 1 − per-user NDCG@10 with gate forced to one | Error definition not supplied |
| Latency | Synced model representation timing, batch one | Not an end-to-end service measurement |

Per-position prefix batches are transparent and straightforward, but slower than a fused batched-user implementation for long sequences. At the reported moderate dataset sizes they may still require substantial GPU time. No paper throughput/training-time claim is made for this code.

## Temporal limits

Snapshot audits verify that consumed personal and neighbor artifacts use training behavior and precede the prediction time. All item descriptors are allowed offline for the catalog, following the paper, but correctness still depends on item text being available independently of held-out behavior.

Static real graphs without edge timestamps cannot establish when each relation became known. A training-only proxy graph likewise may encode training interactions later than an individual historical query. The code records these graphs as static and cannot certify the stronger claim that *every graph-derived input* is historically available. Timestamped real edges are filtered at query time. Recovering exact historical graph snapshots would remove this remaining uncertainty.

The deliberately leakage-prone snapshot-discipline ablation is available only through `leaky_snapshots: true` or `variant --name leaky-snapshots`. It consumes each user’s and neighbor’s final TRAINING snapshot irrespective of commit time. Its audit reports violations and its result is marked `intentional_leakage_ablation: true`. It still does not use held-out interactions to build profiles. The paper does not state which future-snapshot scope was used, so the claimed +3.4% cannot be recreated uniquely. The default pipeline always rejects temporal violations.

## Baseline and experiment coverage

Version 0.3 contains executable local paths for every Table 9 baseline, all principal module/snapshot ablations, fixed-fusion validation tuning, separate/shared alignment projections, sampled/full neighborhoods, weighted/JSON profile embeddings, recency windows, BCE/BPR, single/two-strike drift, committed/live freshness, encoder ARI, proxy graphs, stress comparisons, sweeps, baseline tuning, five-seed aggregation and paired testing. The suite runner executes those paths and retains measured outputs.

The original-baseline adaptations are specified in BASELINES.md. In particular, LLaRA's original release uses candidate-prompt generation, whereas the manuscript claims full-catalog evaluation and an unspecified router. This repository implements hybrid prompts, curriculum and LoRA, with full-catalog conditional likelihood. RLMRec defaults to contrastive semantic alignment on LightGCN and optionally supports masked generative reconstruction. The manuscript does not identify the original RLMRec variant or its semantic prompts.

The ARI implementation uses metadata category strings, KMeans/MiniBatchKMeans and a recorded seed; the original category mapping and clustering setup are unknown. Neighbor sampling defaults to 100 candidates when enabled; its original sample budget is unknown. These explicit choices implement the experiments without pretending to recover missing settings.

The main suite uses preset SUGAR settings from Section 4.4. Original online/offline tuning can be re-executed with the provided search tools; the exact original sampled trials cannot be reconstructed from the paper. Baseline tuning evaluates validation only. Default runs abort on failures; they never silently drop failed seeds. The paper's statement that some Epinions LoRA baselines average only converged seeds conflicts with the general five-seed table description and requires original failure logs.

## Verification performed

See `validation_report.json` for the actual test environment and commands. Unit tests exercise equations, causal masking, trainable/frozen gradients, sampling, split semantics, user/neighbor commit boundaries, stale/corrupted profiles, long pending-job lags, cache consistency, full-catalog ranking and deterministic checkpoint resumption. Synthetic training is not validation of any reported accuracy, statistical significance, GPU latency, semantic quality or L40S cost.


## Version 0.3 coverage additions

Pooled JS threshold robustness at 85/90/95 percentiles, module call/token/time accounting, trainable parameter overhead, actual threshold Pareto fronts, computed Table 4–9 exports, and failed-seed accounting are executable. The default research specification rejects fixture backends and demo inputs. Random corruption samples another user uniformly from temporally available profiles with a recorded seed. Malformed item JSON can fall back to hashtags plus the pinned `en_core_web_sm` 3.7.1 pretrained entity recognizer; its choice is explicit because the original NER model is unidentified. Semantic projection caching is exact-input inference reuse, not a claim that live side information remains constant for a whole snapshot chunk.
