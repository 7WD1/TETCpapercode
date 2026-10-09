"""Stage 4: four query families sharing one circuit, DFA and point cache."""
from itertools import combinations
from math import comb
from types import MappingProxyType
import numpy as np
from .mapping import bits


class QueryRuntime:
    def __init__(self, diagram, automaton, center, *, feature_indices=None,
                 subset_budget=100000, raw_evidence=None):
        if automaton.diagram is not diagram or automaton.order != diagram.order:
            raise ValueError("runtime requires the matching frozen circuit/automaton pair")
        self.diagram, self.automaton = diagram, automaton
        self.center = bits(center, diagram.n)
        self.feature_indices = tuple(range(diagram.n)) if feature_indices is None else tuple(feature_indices)
        if len(self.feature_indices) != diagram.n:
            raise ValueError("feature_indices must match the local dimension")
        self.subset_budget = subset_budget
        evidence = diagram.observations if raw_evidence is None else raw_evidence
        if any(value not in (0, 1) for value in evidence.values()):
            raise ValueError("raw evidence must have binary reconstructed-target labels")
        self.evidence = MappingProxyType({bits(row, diagram.n): int(value) for row, value in evidence.items()})
        self.raw_complete = len(self.evidence) == (1 << diagram.n)
        self.raw_equivalent = self.raw_complete and all(diagram.evaluate(row) == value
                                                       for row, value in self.evidence.items())
        self._points = {}
        self._answers = {}
        self.transition_rows_evaluated = 0
        self.family_requests = 0

    def evaluate_batch(self, assignments):
        rows = [bits(row, self.diagram.n) for row in assignments]
        missing = list(dict.fromkeys(row for row in rows if row not in self._points))
        if missing:
            matrix = np.asarray(missing, dtype=int).reshape(len(missing), self.diagram.n)
            values = self.automaton.run_batch(matrix)
            self._points.update(zip(missing, values.tolist()))
            self.transition_rows_evaluated += len(missing)
        return [self._points[row] for row in rows]

    def _flip(self, subset):
        row = list(self.center)
        for i in subset:
            row[i] ^= 1
        return tuple(row)

    def _point_status(self, rows):
        if any(row in self.evidence and self.evidence[row] != self.diagram.evaluate(row) for row in rows):
            return "known_disagreement"
        if any(row not in self.evidence for row in rows):
            return "unknown"
        return "known_agreement"

    def why(self):
        """All minimum-cardinality sufficient reasons; never label greediness exact."""
        center_value = self.evaluate_batch([self.center])[0]
        n = self.diagram.n
        examined = 0
        ordering = sorted(range(n), key=lambda i: self.feature_indices[i])
        for size in range(n + 1):
            reasons = []
            for subset in combinations(ordering, size):
                examined += 1
                if examined > self.subset_budget:
                    return {"status": "budget_exceeded", "minimum_cardinality": None,
                            "reasons": [], "subset_tests": examined - 1,
                            "compiled_output": center_value, "raw_target_status": "unknown"}
                fixed = {i: self.center[i] for i in subset}
                if self.diagram.implies(fixed, center_value):
                    reasons.append(list(subset))
            if reasons:
                supports = []
                for subset in reasons:
                    consistent = [(row, value) for row, value in self.evidence.items()
                                  if all(row[i] == self.center[i] for i in subset)]
                    complete = len(consistent) == (1 << (n - len(subset)))
                    supports.append("known_agreement" if complete and all(value == center_value
                                    for _, value in consistent) else
                                    "known_disagreement" if any(value != center_value
                                    for _, value in consistent) else "unknown")
                return {"status": "exact_on_compiled_object", "compiled_output": center_value,
                        "minimum_cardinality": size, "reasons": reasons,
                        "canonical_reason": reasons[0],
                        "canonical_original_indices": [self.feature_indices[i] for i in reasons[0]],
                        "subset_tests": examined, "raw_sufficiency_statuses": supports,
                        "raw_minimum_status": "certified" if self.raw_equivalent else "unknown",
                        "raw_target_status": "known_agreement" if self.raw_equivalent else "unknown"}
        raise RuntimeError("fixing every bit must imply the center output")

    def why_not(self):
        """Paper's single-bit overturning set, with observed point evidence."""
        rows = [self.center] + [self._flip((i,)) for i in range(self.diagram.n)]
        values = self.evaluate_batch(rows)
        changed = [i for i, value in enumerate(values[1:]) if value != values[0]]
        return {"status": "exact_on_compiled_object", "center_output": values[0],
                "single_bit_overturning": changed,
                "original_indices": [self.feature_indices[i] for i in changed],
                "flip_outputs": values[1:],
                "raw_flip_values": [self.evidence.get(row) for row in rows[1:]],
                "raw_target_status": self._point_status(rows)}

    def interaction(self):
        pairs = list(combinations(range(self.diagram.n), 2))
        rows = [self.center] + [self._flip((i,)) for i in range(self.diagram.n)]
        rows += [self._flip(pair) for pair in pairs]
        values = self.evaluate_batch(rows)
        baseline, singles = values[0], values[1:1 + self.diagram.n]
        result = []
        for (i, j), joint in zip(pairs, values[1 + self.diagram.n:]):
            required = [self.center, self._flip((i,)), self._flip((j,)), self._flip((i, j))]
            magnitude = abs(joint - baseline - (singles[i] - baseline) - (singles[j] - baseline))
            known = all(row in self.evidence for row in required)
            raw = [self.evidence[row] for row in required] if known else None
            result.append({"features": [i, j],
                           "original_indices": [self.feature_indices[i], self.feature_indices[j]],
                           "magnitude": magnitude, "normalized_magnitude": magnitude / 2,
                           "raw_magnitude": abs(raw[3] - raw[1] - raw[2] + raw[0]) if known else None,
                           "raw_target_status": self._point_status(required)})
        return {"status": "exact_on_compiled_object", "pairs": result,
                "raw_target_status": self._point_status(rows)}

    def sensitivity(self, costs=None):
        """Exact compiled sphere profile and weighted shortest flip, no sampling."""
        costs = np.ones(self.diagram.n) if costs is None else np.asarray(costs, dtype=float)
        profile = self.automaton.disagreement_profile(self.center)
        nearest = self.automaton.nearest_flip(self.center, costs)
        raw_center = self.evidence.get(self.center)
        observed_counts, observed_flips = [0] * len(profile), [0] * len(profile)
        observed_costs = []
        for row, value in self.evidence.items():
            changed = [i for i in range(self.diagram.n) if row[i] != self.center[i]]
            d = len(changed)
            observed_counts[d] += 1
            if raw_center is not None and value != raw_center:
                observed_flips[d] += 1
                observed_costs.append(sum(float(costs[i]) for i in changed))
        for d, item in enumerate(profile):
            unknown = comb(self.diagram.n, d) - observed_counts[d]
            item["raw_unknown_count"] = unknown
            item["raw_rate_bounds"] = ([observed_flips[d] / item["total"],
                                        (observed_flips[d] + unknown) / item["total"]]
                                       if raw_center is not None else [0.0, 1.0])
        hamming = next((item["radius"] for item in profile if item["flips"] > 0), None)
        return {"status": "exact_on_compiled_object", "profile": profile,
                "minimum_hamming_distance": hamming, "no_compiled_flip": hamming is None,
                "weighted_minimum": nearest,
                "raw_observed_cost_upper_bound": min(observed_costs) if observed_costs else None,
                "raw_target_status": "known_agreement" if self.raw_equivalent else
                                     "known_disagreement" if any(self.diagram.evaluate(row) != value
                                     for row, value in self.evidence.items()) else "unknown",
                "raw_minimum_certified": self.raw_equivalent}

    def answer(self, family, *, costs=None):
        if family not in {"why", "why_not", "interaction", "sensitivity"}:
            raise ValueError(f"unsupported query family: {family}")
        self.family_requests += 1
        cost_key = None if costs is None else tuple(float(c) for c in costs)
        key = (family, cost_key if family == "sensitivity" else None)
        if key not in self._answers:
            self._answers[key] = getattr(self, family)(costs=costs) if family == "sensitivity" else getattr(self, family)()
        return self._answers[key]

    def answer_all(self, *, costs=None):
        return {family: self.answer(family, costs=costs) for family in
                ("why", "why_not", "interaction", "sensitivity")}

    def stats(self):
        return {"family_requests": self.family_requests, "cached_family_answers": len(self._answers),
                "cached_point_assignments": len(self._points),
                "transition_rows_evaluated": self.transition_rows_evaluated,
                "shared_objects": 1, "raw_domain_complete": self.raw_complete}
