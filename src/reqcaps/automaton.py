"""Stage 3: frozen-order bridge with explicit skipped-layer debt states."""
from functools import lru_cache
from math import comb
import numpy as np
from .mapping import bits, frozen_array


class Automaton:
    """Layered binary DFA; all paths consume exactly n scheduled bits."""
    def __init__(self, diagram):
        self.diagram = diagram
        self.n = diagram.n
        self.order = diagram.order
        self.state_keys = []
        transitions, accepts, kinds, state_ids = [], [], [], {}

        def allocate(node_id, pending_index):
            key = (node_id, pending_index)
            if key in state_ids:
                return state_ids[key]
            state_id = len(state_ids)
            state_ids[key] = state_id
            self.state_keys.append(key)
            transitions.append([state_id, state_id])
            accepts.append(pending_index == self.n and node_id == 1)
            kinds.append("terminal" if pending_index == self.n else "debt")
            if pending_index == self.n:
                if node_id not in (0, 1):
                    raise RuntimeError("nonterminal residual after the last input")
                return state_id
            if node_id <= 1:
                child = allocate(node_id, pending_index + 1)
                transitions[state_id] = [child, child]
            else:
                node = diagram.nodes[node_id]
                level = diagram.level[node.variable]
                if level < pending_index:
                    raise RuntimeError("BDD order is inconsistent with residual state")
                if level > pending_index:
                    child = allocate(node_id, pending_index + 1)
                    transitions[state_id] = [child, child]
                else:
                    kinds[state_id] = "decision"
                    transitions[state_id] = [allocate(node.low, pending_index + 1),
                                             allocate(node.high, pending_index + 1)]
            return state_id

        self.start = allocate(diagram.root, 0)
        self.state_keys = tuple(self.state_keys)
        self.kinds = tuple(kinds)
        self.transitions = frozen_array(transitions, dtype=np.int64)
        self.accepts = frozen_array(accepts, dtype=bool)

    def run(self, assignment):
        assignment = bits(assignment, self.n)
        state = self.start
        for variable in self.order:
            state = int(self.transitions[state, assignment[variable]])
        return int(self.accepts[state])

    def run_batch(self, assignments):
        array = np.asarray(assignments)
        if array.ndim != 2 or array.shape[1] != self.n or not np.all((array == 0) | (array == 1)):
            raise ValueError(f"expected a binary matrix with {self.n} columns")
        array = array.astype(np.int64, copy=False)
        states = np.full(len(array), self.start, dtype=np.int64)
        for variable in self.order:
            states = self.transitions[states, array[:, variable]]
        return self.accepts[states].astype(int)

    def trace(self, assignment):
        assignment = bits(assignment, self.n)
        state = self.start
        result = []
        for step, variable in enumerate(self.order):
            next_state = int(self.transitions[state, assignment[variable]])
            result.append({"step": step, "variable": variable, "bit": assignment[variable],
                           "state": state, "state_kind": self.kinds[state], "next_state": next_state})
            state = next_state
        return {"steps": result, "terminal": state, "output": int(self.accepts[state])}

    def disagreement_profile(self, center):
        """Exact Hamming sphere counts by integer dynamic programming."""
        center = bits(center, self.n)
        center_value = self.run(center)
        distribution = {(self.start, 0): 1}
        for variable in self.order:
            following = {}
            for (state, distance), count in distribution.items():
                for value in (0, 1):
                    key = (int(self.transitions[state, value]), distance + (value != center[variable]))
                    following[key] = following.get(key, 0) + count
            distribution = following
        counts = [0] * (self.n + 1)
        for (state, distance), count in distribution.items():
            if int(self.accepts[state]) != center_value:
                counts[distance] += count
        return [{"radius": d, "flips": counts[d], "total": comb(self.n, d),
                 "rate": counts[d] / comb(self.n, d)} for d in range(self.n + 1)]

    def nearest_flip(self, center, costs=None):
        """Exact weighted distance and deterministic witness by DAG DP."""
        center = bits(center, self.n)
        costs = np.ones(self.n) if costs is None else np.asarray(costs, dtype=float)
        if costs.shape != (self.n,) or not np.isfinite(costs).all() or np.any(costs <= 0):
            raise ValueError("costs must be positive finite values, one per local feature")
        target = 1 - self.run(center)

        @lru_cache(None)
        def visit(state):
            _, step = self.state_keys[state]
            if step == self.n:
                return (0.0, ()) if int(self.accepts[state]) == target else (float("inf"), ())
            variable = self.order[step]
            candidates = []
            for value in (0, 1):
                downstream, suffix = visit(int(self.transitions[state, value]))
                cost = downstream + (float(costs[variable]) if value != center[variable] else 0.0)
                candidates.append((cost, (value,) + suffix))
            return min(candidates)

        distance, scheduled = visit(self.start)
        if not np.isfinite(distance):
            return {"distance": None, "robust": True, "witness": None}
        witness = [0] * self.n
        for variable, value in zip(self.order, scheduled):
            witness[variable] = value
        return {"distance": distance, "robust": False, "witness": witness}

    def to_dict(self):
        return {"n": self.n, "order": list(self.order), "start": self.start,
                "states": [{"id": i, "residual": key[0], "pending_index": key[1],
                            "kind": self.kinds[i], "accept": bool(self.accepts[i]),
                            "next": self.transitions[i].tolist()}
                           for i, key in enumerate(self.state_keys)]}
