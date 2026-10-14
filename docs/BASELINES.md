# Local baseline implementations

All nine Table 9 baselines now have executable local implementations. They share the mapped data, validation-based checkpoint selection, seed provenance and full-catalog rank evaluator. They are independent reimplementations/adaptations; the SUGAR manuscript did not provide its baseline source commits or exact adaptation settings.

| Model name | Implementation and objective |
|---|---|
| `sasrec` | Causally masked attention; tied ID scoring; BCE/BPR |
| `gru4rec` | Packed chronological prefixes, GRU hidden state, configurable BCE/BPR-Max |
| `caser` | Vertical/horizontal convolutions, user embedding concatenation, item scoring weights and biases |
| `bert4rec` | Bidirectional encoder, 80/10/10 masked-item replacements, Cloze softmax loss; appended MASK at evaluation |
| `lm-feature` | Frozen descriptor embeddings concatenated with IDs and projected into SASRec |
| `socialmf` | Implicit-feedback MF plus directed normalized neighborhood-mean regularization |
| `lightgcn-plus` | Normalized social-graph propagation fused with SASRec through a learned late concatenation |
| `rlmrec` | LightGCN collaborative space and shared semantic projection; contrastive alignment of users, positive items and negative items |
| `llara` | Frozen pretrained SASRec item embeddings, learned two-layer projection, text/ID hybrid prompt curriculum and LoRA-tuned causal language model |

RLMRec-Gen is additionally available through `rlm_variant: generative`: learned node masking, graph propagation and reconstruction through semantic contrastive supervision. RLMRec-Con is the default Table 9 choice because the manuscript does not identify which original RLMRec variant was used. The CF graph uses only training interactions, never validation/test events.

## LLaRA's necessary protocol adaptation

The original LLaRA release generates a recommended item from a candidate prompt and uses its own candidate-level evaluation. The supplied SUGAR manuscript instead reports full-catalog HR/NDCG and describes a routing mechanism without specifying an implementation. This code implements the verifiable hybrid-prompt/curriculum mechanism from LLaRA and evaluates every item through conditional language-model likelihood.

The training response contains the mapped item ID plus item text. A candidate is an answer label, not included in the observed history. Its likelihood is computed with teacher forcing; all catalog items receive scores. IDs disambiguate duplicate texts. Summed log-likelihood is default; length normalization is configurable. The curriculum starts with text-only histories and gradually inserts projected, frozen sequential ID embeddings. Inference uses hybrid prompts. These choices are adaptations and cannot establish exact equality with the manuscript's unspecified LLaRA run.

`llara_backbone_checkpoint` must be a local SASRec best.pt from the same processed dataset. The frozen SASRec width is recovered from that checkpoint. The pretrained LLM is resolved by an immutable Hugging Face revision; the default uses Qwen2.5-7B-Instruct, while `llara_llm_model` can select Llama-2-7b-hf for an original-model adaptation with appropriate access. The paper's actual baseline LLM is unknown.

LoRA targets attention and feedforward linear modules. Frozen base model weights are not duplicated in checkpoints: only adapter/projector parameters are saved. Preserve the immutable LLM revision, backbone file/hash, tokenizer, configuration and data to restore a run. The source snapshot used to check the original mechanism is recorded in SOURCES.md.

Full-catalog language-model likelihood is expensive: no sampled shortlist, silent distillation or smaller LLM is substituted. `llara_score_batch` controls memory. The `tiny` backend is a frozen random GRU decoder with a small trainable adapter for infrastructure tests, always marked synthetic; it is not a Qwen/Llama model and cannot count as baseline accuracy evidence.

## Static RLMRec semantics

Run `sugar static-profiles` after offline preprocessing. It builds each user's baseline profile from all TRAINING events and retains the prompt/output cache. Main SUGAR snapshots remain temporally joined separately. Alternatively, `rlm_user_embeddings` and `rlm_item_embeddings` accept original baseline caches with matching mapped IDs, row-zero padding, and common semantic dimension.

Using Qwen/BGE structured descriptors/profiles adapts the baseline's semantic-generation pipeline. Original RLMRec's own prompts/embedding-model artifacts can replace those inputs. Only supplying the originals permits a claim that the same semantic representations were used.

## Configuration and tuning

Generate a model configuration:

```bash
sugar variant --config configs/resolved-yelp.yaml --name bert4rec --out configs/yelp-bert4rec.yaml
```

Equivalent names are accepted for all models; `rlmrec-gen` selects generative alignment. The suite performs the same configured number of baseline search trials, chooses by validation only, then runs the selected configuration with each seed. Search spaces are in `suite.py`, and every sampled configuration is saved. The actual configurations sampled in the original experiment remain unknown.

The common user-prefix protocol is an adaptation of GRU4Rec's session training and BERT4Rec's per-user Cloze sampling. SocialMF adapts explicit-rating MF to the paper's uniform implicit feedback. LightGCN+ is a manuscript-specific model whose exact fusion and graph settings were not supplied. These differences are explicit rather than hidden behind the method names.

`export-protocol` and `import-baseline` remain available to cross-check an independent original implementation. Import validates declared protocol metadata and all test-user targets/rank bounds; it does not independently prove an external runner's compliance. Local execution does not depend on any imported scores.
