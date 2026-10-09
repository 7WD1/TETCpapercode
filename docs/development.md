# Development and verification

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q src tests
python -m reqcaps --help
```

The correctness suite uses small truth tables and independent exhaustive oracles. It checks BDD semantics, variable sifting, reordered/debt automata, exact Hamming counts, weighted path optimality, minimum sufficient reasons (including empty and tied reasons), point-cache sharing, incomplete/held-out evidence, lossy centers, numeric validity, deterministic sampling and quiet CLI operation.

No benchmark datasets, training tasks, figure generation, performance measurements or statistical comparisons are run. The repository intentionally excludes generated result directories and model/data binary files.

## Complexity and budgets

Screening uses at most `1 + d*M0` predictor rows before input-cache savings. Truth reconstruction needs one row per unique queried Boolean code before the same cache savings. An exhaustive cube requires `2**n` codes; the enumeration cutoff bounds memory.

ROBDD size is worst-case exponential in the retained dimension. This implementation builds through recursive positive-minterm partitioning, rather than a native apply-based CUDD kernel. Sifting performs repeated rebuilds and is appropriate for modest retained dimensions. A `node_budget` applies to each candidate; failure of the initial build aborts explicitly. Candidates exceeding the budget during sifting are skipped.

For BDD size m, one fixed-subset cofactor test visits at most m nodes. Searching all minimum sufficient reasons can require exponentially many subsets and outputs, bounded here by `why_subset_budget`. This is different from checking a supplied sufficient condition.

One automaton path takes n transitions. Batch execution is O(batch*n). Skipped-layer padding can introduce O(n*m) states in the worst case. Exact sensitivity counting takes O(n*states) dynamic-programming entries and uses Python integers for exact counts; arithmetic cost grows with integer bit length. Weighted nearest flips visit each reachable state once. WHY-NOT and all pairwise interactions use O(n**2) assignments with cache reuse.

The reference implementation makes no claimed wall-clock relationship to the paper's CUDD/CPU setup. Different backends, environment settings, construction splits and variable orders require separate measurement.

## Releasing a change

Run the correctness suite after a semantic change. Preserve unknown-value distinctions, coverage metadata and review flags. Keep experiment output separate from method implementation, and do not check in private datasets or fitted models. If a future experiment harness is added, it must identify dataset versions, preprocessing, frozen folds, model/hyperparameter selection and timing procedures before claiming reproduced paper results.

