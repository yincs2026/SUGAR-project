# Implementation source checks

The uploaded Word manuscript is the model specification. Its SHA-256 is retained in `paper_reference.json`. The original Word file is not included in the source archive.

The following primary documentation was checked to ground API/embedding choices:

- [vLLM 0.5.0 OpenAI-compatible server](https://docs.vllm.ai/en/v0.5.0/serving/openai_compatible_server.html): the legacy `guided_json` request parameter.
- [vLLM 0.5.5 server documentation](https://docs.vllm.ai/en/v0.5.5/serving/openai_compatible_server.html): server API and guided decoding.
- [BAAI BGE-base-en-v1.5 model card](https://huggingface.co/BAAI/bge-base-en-v1.5): frozen sentence embeddings and supported SentenceTransformer usage.
- [BGE pooling configuration](https://huggingface.co/BAAI/bge-base-en-v1.5/blob/main/1_Pooling/config.json): CLS pooling in the official SentenceTransformer checkpoint.

Checked on 2026-10-14. These documentation checks do not constitute an actual vLLM/Qwen/BGE execution. Runtime model commits must be independently resolved and retained as explained in README.md.

## Baseline mechanism checks for version 0.2

Primary original repositories/papers:

- [GRU4Rec official PyTorch implementation](https://github.com/hidasib/GRU4Rec_PyTorch_Official): recurrent/session architecture and BPR-Max option.
- [Caser author implementation](https://github.com/graytowne/caser_pytorch): vertical/horizontal convolutions and user representations.
- [BERT4Rec paper](https://arxiv.org/abs/1904.06690): bidirectional Cloze training and masked evaluation.
- [LightGCN paper](https://arxiv.org/abs/2002.02126): normalized graph propagation.
- [SocialMF original publication](https://doi.org/10.1145/1864708.1864736): neighborhood-mean social regularization; the manuscript's implicit-feedback adaptation is unspecified.
- [RLMRec original repository](https://github.com/HKUDS/RLMRec), mechanism inspected at commit `22413752246de3dee8ab0d509f7f7a8889080f95`: contrastive shared semantic projection and generative masked reconstruction.
- [LLaRA original repository](https://github.com/ljy0ustc/LLaRA), mechanism inspected at commit `9733f732399e95e236f4f2e1d1139c1f0437090e`: frozen recommender item embeddings, projection into language-model tokens, LoRA and curriculum.

Source code was inspected for algorithmic grounding, not copied into this source distribution. Local implementations are independent adaptations to the common input/evaluation contract. Original source-repository licenses remain applicable to separately obtained original software. The pinned inspection commits establish what was checked, not which commits the SUGAR manuscript used.


## Entity fallback

[spaCy English model releases](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.7.1) provide the actual pretrained 3.7.1 NER checkpoint used by the optional malformed-item fallback. Runtime verifies the checkpoint metadata version. This is a concrete choice; the manuscript does not identify its entity extractor.
