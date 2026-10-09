"""Stage 2: reduced ordered BDD for the positive construction minterms.

This portable implementation uses ordinary edges, not complemented pointers.
Reduction shares identical triples and removes nodes with equal children.
Sifting rebuilds the same function under candidate variable orders.
"""
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from .mapping import bits


class NodeBudgetExceeded(RuntimeError):
    """Construction stopped; no approximate diagram is substituted."""


@dataclass(frozen=True)
class Node:
    variable: int
    low: int
    high: int


class ROBDD:
    FALSE = 0
    TRUE = 1

    def __init__(self, n, order, root, nodes, observations, ordering_log=()):
        self.n = n
        self.order = tuple(order)
        self.level = MappingProxyType({var: i for i, var in enumerate(self.order)})
        self.root = root
        self.nodes = MappingProxyType(dict(nodes))
        self.observations = MappingProxyType(dict(observations))
        self.ordering_log = tuple(ordering_log)

    @classmethod
    def compile(cls, assignments, labels, *, n=None, order=None, node_budget=50000,
                sifting_passes=0):
        assignments = list(assignments)
        labels = list(labels)
        if n is None:
            if not assignments:
                raise ValueError("n is required for an empty construction set")
            n = len(assignments[0])
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if not isinstance(node_budget, int) or node_budget < 2:
            raise ValueError("node_budget must allow both terminals")
        if len(assignments) != len(labels):
            raise ValueError("assignments and labels must have equal lengths")
        observations = {}
        for row, label in zip(assignments, labels):
            row = bits(row, n)
            if label not in (0, 1):
                raise ValueError("construction labels must be binary")
            value = int(label)
            if row in observations and observations[row] != value:
                raise ValueError("conflicting labels for the same Boolean code")
            observations[row] = value
        order = tuple(range(n)) if order is None else tuple(order)
        if sorted(order) != list(range(n)):
            raise ValueError("order must be a permutation of local feature indices")
        positives = frozenset(row for row, value in observations.items() if value == 1)
        current = cls._build(n, order, positives, observations, node_budget)
        log = [{"order": list(order), "nodes": current.node_count}]
        for _ in range(sifting_passes):
            changed = False
            for variable in range(n):
                reduced = [v for v in current.order if v != variable]
                best = current
                for position in range(n):
                    candidate_order = tuple(reduced[:position] + [variable] + reduced[position:])
                    if candidate_order == current.order:
                        continue
                    try:
                        candidate = cls._build(n, candidate_order, positives, observations, node_budget)
                    except NodeBudgetExceeded:
                        continue
                    # Preserve the existing order for equal size; deterministic and no oscillation.
                    if candidate.node_count < best.node_count:
                        best = candidate
                if best.order != current.order:
                    changed = True
                    current = best
                    log.append({"order": list(current.order), "nodes": current.node_count})
            if not changed:
                break
        current.ordering_log = tuple(log)
        return current

    @classmethod
    def _build(cls, n, order, positives, observations, node_budget):
        unique, nodes = {}, {}

        def descend(rows, level):
            if not rows:
                return 0
            if len(rows) == (1 << (n - level)):
                return 1
            variable = order[level]
            low_rows = frozenset(row for row in rows if row[variable] == 0)
            high_rows = rows - low_rows
            low, high = descend(low_rows, level + 1), descend(high_rows, level + 1)
            if low == high:
                return low
            key = (variable, low, high)
            if key not in unique:
                if len(nodes) + 2 >= node_budget:
                    raise NodeBudgetExceeded(f"ROBDD exceeds node budget {node_budget}")
                node_id = len(nodes) + 2
                unique[key] = node_id
                nodes[node_id] = Node(*key)
            return unique[key]

        root = descend(positives, 0)
        return cls(n, order, root, nodes, observations)

    @property
    def node_count(self):
        return len(self.nodes) + 2

    @property
    def complete(self):
        return len(self.observations) == (1 << self.n)

    def evaluate(self, assignment):
        assignment = bits(assignment, self.n)
        node_id = self.root
        while node_id > 1:
            node = self.nodes[node_id]
            node_id = node.high if assignment[node.variable] else node.low
        return node_id

    def observed_value(self, assignment):
        """None means unknown f_x; a closed-world circuit 0 does not label it."""
        return self.observations.get(bits(assignment, self.n))

    def implies(self, fixed, output=1):
        """Exact universal cofactor test, including the empty condition."""
        if output not in (0, 1):
            raise ValueError("output must be binary")
        if any(i not in range(self.n) or value not in (0, 1) for i, value in fixed.items()):
            raise ValueError("invalid fixed literal")

        @lru_cache(None)
        def visit(node_id):
            if node_id <= 1:
                return node_id == output
            node = self.nodes[node_id]
            if node.variable in fixed:
                return visit(node.high if fixed[node.variable] else node.low)
            return visit(node.low) and visit(node.high)

        return visit(self.root)

    def to_dict(self):
        return {"n": self.n, "order": list(self.order), "root": self.root,
                "node_count": self.node_count, "complete_construction": self.complete,
                "construction_count": len(self.observations),
                "ordering_log": list(self.ordering_log),
                "nodes": {str(i): {"variable": v.variable, "low": v.low, "high": v.high}
                          for i, v in self.nodes.items()}}

