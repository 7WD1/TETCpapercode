"""Small exact oracles for implementation correctness, not paper experiments."""
from itertools import product, combinations
from math import comb
import numpy as np
import pytest
from reqcaps import ROBDD, Automaton, QueryRuntime, NodeBudgetExceeded


def cube(n):
    return list(product((0, 1), repeat=n))


@pytest.mark.parametrize("n", range(6))
def test_random_truth_tables_bridge_and_profile(n):
    rng = np.random.default_rng(100 + n)
    rows = cube(n)
    for _ in range(12):
        labels = rng.integers(0, 2, len(rows)).tolist()
        order = rng.permutation(n).tolist()
        diagram = ROBDD.compile(rows, labels, n=n, order=order, sifting_passes=1)
        machine = Automaton(diagram)
        matrix = np.asarray(rows, dtype=int).reshape(len(rows), n)
        assert machine.run_batch(matrix).tolist() == labels
        assert [diagram.evaluate(row) for row in rows] == labels
        for center in rows[:4]:
            profile = machine.disagreement_profile(center)
            central = diagram.evaluate(center)
            for radius, entry in enumerate(profile):
                sphere = [row for row in rows if sum(a != b for a, b in zip(row, center)) == radius]
                assert entry["total"] == comb(n, radius) == len(sphere)
                assert entry["flips"] == sum(diagram.evaluate(row) != central for row in sphere)
            weights = rng.uniform(.1, 3, n)
            candidate_costs = [sum(weights[i] for i in range(n) if row[i] != center[i])
                               for row in rows if diagram.evaluate(row) != central]
            nearest = machine.nearest_flip(center, weights)
            if candidate_costs:
                assert nearest["distance"] == pytest.approx(min(candidate_costs))
                assert machine.run(nearest["witness"]) != central
            else:
                assert nearest == {"distance": None, "robust": True, "witness": None}


def test_skipped_layers_reordered_input_and_trace():
    rows = cube(5)
    labels = [row[3] for row in rows]
    diagram = ROBDD.compile(rows, labels, order=[4, 0, 3, 1, 2])
    machine = Automaton(diagram)
    assert "debt" in machine.kinds
    for row, label in zip(rows, labels):
        trace = machine.trace(row)
        assert trace["output"] == label
        assert [step["variable"] for step in trace["steps"]] == [4, 0, 3, 1, 2]
        assert len(trace["steps"]) == 5
    assert not machine.transitions.flags.writeable


def test_sifting_reduces_mux_and_preserves_function():
    rows = cube(3)
    labels = [row[1] if row[0] else row[2] for row in rows]
    unsifted = ROBDD.compile(rows, labels, order=[1, 2, 0])
    sifted = ROBDD.compile(rows, labels, order=[1, 2, 0], sifting_passes=2)
    assert sifted.node_count < unsifted.node_count
    assert [sifted.evaluate(row) for row in rows] == labels


def test_incomplete_all_positive_never_becomes_constant_true():
    diagram = ROBDD.compile([(1, 1)], [1], n=2)
    assert diagram.evaluate((1, 1)) == 1
    assert diagram.evaluate((0, 1)) == 0
    assert diagram.observed_value((0, 1)) is None
    runtime = QueryRuntime(diagram, Automaton(diagram), (1, 1))
    assert runtime.why_not()["raw_target_status"] == "unknown"
    answer = runtime.sensitivity()
    assert not answer["raw_minimum_certified"]
    assert answer["profile"][1]["raw_rate_bounds"] == [0, 1]
    assert runtime.why()["raw_minimum_status"] == "unknown"


def test_why_minimum_all_ties_and_empty_constant_reason():
    rows = cube(3)
    labels = [int(row[0] or (row[1] and row[2])) for row in rows]
    diagram = ROBDD.compile(rows, labels)
    runtime = QueryRuntime(diagram, Automaton(diagram), (1, 1, 1))
    assert runtime.why()["reasons"] == [[0]]
    assert runtime.why()["raw_minimum_status"] == "certified"
    tie = ROBDD.compile(cube(2), [int(a or b) for a, b in cube(2)])
    assert QueryRuntime(tie, Automaton(tie), (1, 1)).why()["reasons"] == [[0], [1]]
    for value in (0, 1):
        constant = ROBDD.compile(cube(2), [value] * 4)
        constant_runtime = QueryRuntime(constant, Automaton(constant), (0, 1))
        assert constant_runtime.why()["reasons"] == [[]]
        assert constant_runtime.sensitivity()["no_compiled_flip"]


def test_why_against_bruteforce_minimum_not_greedy():
    rng = np.random.default_rng(81)
    rows = cube(4)
    for _ in range(15):
        labels = rng.integers(0, 2, len(rows)).tolist()
        diagram = ROBDD.compile(rows, labels)
        center = (1, 1, 1, 1)
        baseline = diagram.evaluate(center)
        oracle = []
        for size in range(5):
            for subset in combinations(range(4), size):
                if all(diagram.evaluate(row) == baseline for row in rows
                       if all(row[i] == center[i] for i in subset)):
                    oracle.append(list(subset))
            if oracle:
                break
        assert QueryRuntime(diagram, Automaton(diagram), center).why()["reasons"] == oracle


def test_interaction_xor_and_cache_shared_across_families():
    rows = cube(2)
    diagram = ROBDD.compile(rows, [a ^ b for a, b in rows])
    runtime = QueryRuntime(diagram, Automaton(diagram), (1, 0))
    assert runtime.answer("why_not")["single_bit_overturning"] == [0, 1]
    assert runtime.transition_rows_evaluated == 3
    assert runtime.answer("interaction")["pairs"][0]["magnitude"] == 2
    assert runtime.transition_rows_evaluated == 4
    runtime.answer_all()
    runtime.answer_all()
    assert runtime.transition_rows_evaluated == 4
    assert runtime.stats()["cached_family_answers"] == 4


def test_budget_and_invalid_inputs_fail_visibly():
    rows = cube(4)
    labels = [sum(row) % 2 for row in rows]
    with pytest.raises(NodeBudgetExceeded):
        ROBDD.compile(rows, labels, node_budget=3)
    diagram = ROBDD.compile(rows, labels)
    answer = QueryRuntime(diagram, Automaton(diagram), (1, 0, 0, 0), subset_budget=1).why()
    assert answer["status"] == "budget_exceeded"
    assert answer["minimum_cardinality"] is None
    with pytest.raises(ValueError, match="conflicting"):
        ROBDD.compile([(0,), (0,)], [0, 1])
    with pytest.raises(ValueError):
        diagram.evaluate([.5, 0, 0, 1])
    with pytest.raises(ValueError):
        Automaton(diagram).nearest_flip((0, 0, 0, 0), [1, 1, 0, 1])


def test_holdout_evidence_cannot_certify_closed_world_as_raw_target():
    diagram = ROBDD.compile([(1, 1)], [1], n=2)
    evidence = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    runtime = QueryRuntime(diagram, Automaton(diagram), (1, 1), raw_evidence=evidence)
    assert runtime.why_not()["raw_target_status"] == "known_disagreement"
    assert runtime.sensitivity()["raw_target_status"] == "known_disagreement"
    assert not runtime.sensitivity()["raw_minimum_certified"]

