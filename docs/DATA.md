# Data contract

Raw files are UTF-8. IDs are strings and must agree across files. Timestamps are numeric seconds or ISO 8601; timezone-free timestamps are interpreted as UTC. Preserve genuine timezone semantics during source conversion.

Example interactions:

```csv
user_id,item_id,timestamp,rating
u1,i1,2023-01-01T10:00:00Z,5
u1,i2,2023-01-02T10:00:00Z,4
```

Example item:

```json
{"item_id":"i1","text":"A book about badminton training","category":"sport","available_at":"2022-12-01T00:00:00Z"}
```

Example directed edge (u1 consumes u2's neighborhood signal):

```csv
user_id,neighbor_id,timestamp
u1,u2,2022-12-01T00:00:00Z
```

A missing item availability time means catalog content is assumed pre-existing; a missing edge timestamp means a static graph assumption. These assumptions must be verified for the particular source. Categories must come from independent item metadata for JS drift analysis; unknown categories permit recommendation training but are insufficient for detector evaluation.

The preprocessor maps sorted raw user/item IDs to consecutive integers starting at one. Zero is reserved for padding. Duplicates in item metadata are rejected; repeated interactions remain separate events. A genuine iterative core is computed on event counts, before per-user leave-one-out splitting. The removed events and the original raw files should be retained privately for provenance; source file hashes are recorded.

`data.json` includes catalog, graph, sequences, original-to-mapped dictionaries, counts and a content fingerprint. The final two sequence events are validation and test. No M2/M3 artifact incorporates either event. M4's recent vector also excludes them. M5's test prefix includes the validation event. Neither validation nor test targets influence training losses.

Snapshots carry checkpoint/source positions, source timestamps, commit positions/timestamps, profile JSON, embedding, evidence indices/counts, freshness and recent vectors. A commit beyond the last training event has a null timestamp and is never consumed. This deliberately prevents using a held-out event to make a queued update visible.

No processed benchmark is supplied. The exact paper statistics must be compared against the counts produced from the author's dataset releases, never imposed through hardcoded subsampling.
