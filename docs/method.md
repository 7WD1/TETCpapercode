# Method-to-code map

The implementation follows the manuscript's four-stage decomposition. All model calls occur during the build; query execution consumes the frozen local object.

## 1. Local function mapping

`build_mapping` evaluates `predict(x)` and estimates each coordinate's label-change probability by substituting values from a seeded background draw. TopK ties use ascending original feature index. Retained coordinates use the background median and the conditional means below and above that threshold. Empty threshold sides and equal representatives are dropped without silently introducing an arbitrary binary feature. Unretained coordinates remain equal to the audited input.

`LocalMapping.encode` implements beta, and `reconstruct` implements rho. Reconstruction is coordinate-preserving in processed numeric space. For retained nondegenerate features, `encode(reconstruct(v)) == v`; `reconstruct(encode(x)) == x` is **not** required and usually does not hold. Its prediction can differ from `predict(x)`, which is explicitly flagged.

Exhaustive construction uses the complete cube up to the enumeration limit. Beyond it, sampling combines a bounded Hamming neighborhood and a random supplement outside that neighborhood. A separate seed-controlled disjoint set is queried for audit. A nonzero `holdout_fraction` instead also removes sampled rows from construction, always retaining the center.

## 2. Compile

`ROBDD.compile` implements the positive construction-minterm function. The recursion partitions minterms by the current variable; identical `(variable, low, high)` triples share a node and equal children eliminate the decision. Empty positive sets produce false. A true terminal is valid only when every assignment in that subcube has a positive minterm; an incomplete all-positive sample never collapses to an unjustified full-domain true.

The unique table uses ordinary edges. There is no complemented-edge parity bookkeeping. The optional sifting pass places each variable at every candidate position and rebuilds the same minterm function. Strict node-count improvements are retained; ties preserve the current order. The final permutation is frozen. This is a portable Rudell-style strategy, not a CUDD binding.

Node budgets fail with `NodeBudgetExceeded`. No truncated graph or sampled approximation is returned under an exact result label.

## 3. Bridge

Each reachable DFA state identifies `(BDD residual node, pending schedule index)`. At a decision layer its 0/1 transitions follow the node's children. Before a skipped layer, both transitions advance the schedule index without changing the residual. Terminal residuals also consume all remaining scheduled bits before reaching accepting/rejecting terminal states. Thus every path consumes exactly n bits in `diagram.order`.

`run_batch` vectorizes transitions across rows of one local object, and `trace` records decision/debt states, variables and transitions. Queries accept vectors in local feature-index order; the automaton internally reorders the reads using the frozen schedule.

## 4. Unified queries

- **WHY:** Enumerate subsets in increasing cardinality; use exact BDD universal-cofactor checks. Return the complete first successful cardinality layer and a canonical original-feature tie-break. A budget exhausted partway through a layer returns `budget_exceeded`, not a partial exact argmin.
- **WHY-NOT:** Evaluate all single-bit flips in one batch, compare with the compiled center, and map affected local indices to original feature indices.
- **INTERACTION:** Reuse cached center and single-flip values, evaluate joint flips, and compute the manuscript's absolute Boolean second difference.
- **SENSITIVITY:** Propagate integer path counts indexed by Hamming distance; divide by exact sphere cardinalities. Solve weighted nearest opposite-output paths using DAG dynamic programming, including skipped variables and strictly positive costs.

The raw evidence map stores observations on construction and hold-out codes without calling unknown labels zero. Point operators display evidence support. The sensitivity profile additionally bounds the true reconstructed-target disagreement rate using observed flips and unknown sphere members. These bounds are deterministic coverage bounds, not confidence intervals.

If the compiled center is false, operators preserve/contrast that compiled output and the build flags the disagreement with the originally audited target. That diagnostic result does not justify the original target prediction.

## Sharing

An `AuditArtifact` stores one `LocalMapping`, `ROBDD`, `Automaton`, `QueryRuntime` and audit record. Repeated query requests reuse the pair, point-value memo and family-answer memo. Costs are part of the sensitivity cache key. Separate instances are constructed separately; the implementation does not assume their reconstruction functions or variable orders are identical.

