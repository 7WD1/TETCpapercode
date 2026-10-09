# API and artifact schema

```python
from reqcaps import Config, ReQCAPS

engine = ReQCAPS(Config(seed=42, ordering="sift"))
artifact = engine.build(model, x, X_background, feature_names=names)
answers = artifact.answer_all()

# Optional processed-domain predicate. It returns one bool per row.
artifact = engine.build(model, x, X_background,
                        validity=lambda X: ((X >= 0) & (X <= 1)).all(axis=1))
```

`model` exposes `predict(X)` or is callable. Inputs are finite numeric vectors/matrices. Preprocessing and model training occur outside this package; use the exact feature transformation expected by the fitted predictor. For a scikit-learn pipeline receiving raw nonnumeric inputs, supply a callable wrapper accepting the processed numeric representation instead.

`feature_names` maps processed indices to displayed names. Costs use the **retained local-feature order**, not the original full dimension. Selected original indices are available as `artifact.mapping.selected`.

```python
artifacts = engine.build_batch(model, X_audit, X_background)
requests = [(0, "why"), (0, "interaction"), (1, "why_not"), (0, "sensitivity")]
responses = engine.answer_batch(artifacts, requests,
                                costs={0: local_positive_costs})
```

This phase makes no classifier calls and performs no new compilation or bridging. It processes requests in order and returns one response per request. Data-parallel DFA transitions are vectorized; the instance build loop is sequential.

## Inspection

`artifact.to_dict()` returns a JSON-compatible dictionary with:

| Field | Meaning |
| --- | --- |
| `schema_version` | Current value: 1 |
| `target_label` | Original audited classifier label |
| `scope` | `finite_processed_feature_reconstruction` |
| `compiled_semantics` | `positive_construction_minterms_closed_world` |
| `mapping` | Retained/dropped indices, representatives, thresholds, importance and center code |
| `audit` | Construction/hold-out coverage, actual checks, review flags, separately counted black-box rows |
| `config` | Frozen construction/query budgets and seed |
| `answers` | Four compiled queries and reconstructed-target evidence statuses |
| `sharing` | Cache entries, family requests, actual transition rows, raw-domain coverage |
| `circuit` | Nodes, root, variable order, construction coverage and ordering history |
| `automaton` | State residuals, pending indices, debt/decision/terminal kinds and transitions |

`include_structure=False` omits graph/state tables. Serialization has no side effect: writing the returned object is the caller's choice. A no-flip distance is `null` with an explicit robustness flag; JSON infinity and NaN are not emitted.

## Status fields

- `exact_on_compiled_object` certifies the stated finite-circuit computation.
- `budget_exceeded` means the exact requested WHY answer was not completed.
- `known_agreement` means all needed observed reconstructed-target values match the compiled ones, or full equivalence is required and established.
- `known_disagreement` exposes an observed mismatch.
- `unknown` means evidence is insufficient for a target-function certificate.

`raw_sufficiency_statuses` can establish a particular sufficient reason on a fully observed cofactor even when its **minimum** status on the target function is unknown. `raw_rate_bounds` state observed coverage bounds per Hamming sphere. A hold-out score evaluates rho-reconstructed codes, not arbitrary unseen real inputs.

WHY reports a known reconstructed-target disagreement before an unknown equivalence status, including when its subset budget expires. Its `raw_disagreement_witness` identifies an observed code and the differing compiled/raw outputs; this witness establishes disagreement with the compiled object, while the per-reason support statuses describe the sufficient cofactors. Weighted sensitivity raises `OverflowError` if a reachable minimum cost exceeds the finite floating-point range, keeping overflow separate from an unreachable flip. An incomplete construction with an empty hold-out uses `audit.holdout_interpretation: not_available_incomplete_construction`.

Review `audit.requires_review` and `audit.flags` before interpreting answers as evidence about the audited classifier. Direct CLI execution is quiet and leaves those flags available in an explicitly requested JSON artifact.

