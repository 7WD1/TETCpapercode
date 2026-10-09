"""Stage 1: background substitution, frozen Booleanization and reconstruction."""
from dataclasses import dataclass
from itertools import combinations, product
from math import comb
import numpy as np


def bits(value, n):
    """Validate binary coordinates without silently rounding real values."""
    raw = np.asarray(value)
    if raw.shape != (n,) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"expected a binary vector of length {n}")
    return tuple(int(v) for v in raw)


def frozen_array(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def validate_domain(predicate, values, description):
    if predicate is None:
        return
    result = np.asarray(predicate(values))
    if result.shape != (len(values),) or result.dtype.kind != "b":
        raise ValueError("validity(X) must return one boolean per processed row")
    if not result.all():
        raise ValueError(f"{description} violates validity predicate")


class BlackBox:
    """Label-only classifier adapter; count evaluated rows, deduplicate inputs."""
    def __init__(self, model):
        self.function = model.predict if hasattr(model, "predict") else model
        if not callable(self.function):
            raise TypeError("model must be callable or expose predict(X)")
        self.cache = {}
        self.query_count = 0

    def predict(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("model inputs must be a finite 2D numeric matrix")
        keys = [tuple(row) for row in array]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            labels = np.asarray(self.function(np.asarray(missing, dtype=float)))
            if labels.shape != (len(missing),):
                raise ValueError("predict(X) must return one scalar label per row")
            if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
                raise ValueError("predict(X) returned a nonfinite label")
            self.query_count += len(missing)
            self.cache.update(zip(missing, labels.tolist()))
        return np.asarray([self.cache[key] for key in keys])


@dataclass(frozen=True)
class LocalMapping:
    anchor: np.ndarray
    selected: tuple
    thresholds: np.ndarray
    low: np.ndarray
    high: np.ndarray
    importance: np.ndarray
    dropped: tuple

    @property
    def n(self):
        return len(self.selected)

    def encode(self, instance):
        instance = np.asarray(instance, dtype=float)
        if instance.shape != self.anchor.shape or not np.isfinite(instance).all():
            raise ValueError("instance shape must match the anchor")
        return tuple(int(v) for v in instance[list(self.selected)] >= self.thresholds)

    def reconstruct(self, assignment):
        assignment = bits(assignment, self.n)
        value = np.array(self.anchor, copy=True)
        value[list(self.selected)] = np.where(assignment, self.high, self.low)
        return value

    def reconstruct_batch(self, assignments):
        rows = [self.reconstruct(row) for row in assignments]
        return np.asarray(rows, dtype=float).reshape(-1, len(self.anchor))

    def to_dict(self):
        return {"selected_indices": list(self.selected), "dropped_indices": list(self.dropped),
                "thresholds": self.thresholds.tolist(), "low": self.low.tolist(),
                "high": self.high.tolist(), "importance": self.importance.tolist(),
                "center": list(self.encode(self.anchor))}


def build_mapping(model, anchor, background, config, rng, validity=None):
    anchor = np.asarray(anchor, dtype=float)
    background = np.asarray(background, dtype=float)
    if anchor.ndim != 1 or not len(anchor) or background.ndim != 2:
        raise ValueError("anchor must be a nonempty vector and background a matrix")
    if background.shape[1] != len(anchor) or not len(background):
        raise ValueError("background must have rows and match anchor dimensions")
    if not np.isfinite(anchor).all() or not np.isfinite(background).all():
        raise ValueError("preprocess missing/nonfinite values before auditing")
    validate_domain(validity, anchor.reshape(1, -1), "anchor")
    target = model.predict(anchor.reshape(1, -1))[0]
    count = min(config.background_samples, len(background))
    importance = np.zeros(len(anchor))
    for i in range(len(anchor)):
        reference = rng.choice(len(background), count, replace=False)
        substituted = np.tile(anchor, (count, 1))
        substituted[:, i] = background[reference, i]
        validate_domain(validity, substituted, "background substitution")
        importance[i] = np.mean(model.predict(substituted) != target)
    ranked = sorted(range(len(anchor)), key=lambda i: (-importance[i], i))
    retained, dropped, thresholds, low, high = [], [], [], [], []
    for i in sorted(ranked[:min(config.feature_budget, len(anchor))]):
        column = background[:, i]
        threshold = float(np.median(column))
        below, above = column[column < threshold], column[column >= threshold]
        if not len(below) or not len(above):
            dropped.append(i)
            continue
        low_value, high_value = float(np.mean(below)), float(np.mean(above))
        if low_value == high_value or not (low_value < threshold <= high_value):
            dropped.append(i)
            continue
        retained.append(i)
        thresholds.append(threshold)
        low.append(low_value)
        high.append(high_value)
    return LocalMapping(frozen_array(anchor), tuple(retained), frozen_array(thresholds),
                        frozen_array(low), frozen_array(high), frozen_array(importance),
                        tuple(dropped)), target


def sample_construction(center, config, rng):
    """Full Boolean cube or a budgeted Hamming ball and far-field supplement."""
    n = len(center)
    if n <= config.enumeration_limit:
        return list(product((0, 1), repeat=n)), {"strategy": "exhaustive", "near_complete": True}
    radius = min(config.hamming_radius, n)
    ball_size = sum(comb(n, d) for d in range(radius + 1))
    near = {tuple(center)}
    if ball_size <= config.near_budget:
        for d in range(1, radius + 1):
            for subset in combinations(range(n), d):
                value = list(center)
                for j in subset:
                    value[j] ^= 1
                near.add(tuple(value))
    elif radius:
        # Radius classes are selected in proportion to their sphere cardinality.
        probabilities = np.asarray([comb(n, d) for d in range(1, radius + 1)], dtype=float)
        probabilities /= probabilities.sum()
        for _ in range(config.near_budget * 20):
            if len(near) >= config.near_budget:
                break
            d = int(rng.choice(np.arange(1, radius + 1), p=probabilities))
            value = list(center)
            for j in rng.choice(n, d, replace=False):
                value[j] ^= 1
            near.add(tuple(value))
    construction = set(near)
    for _ in range(config.far_budget * 30):
        if len(construction) >= len(near) + config.far_budget:
            break
        row = tuple(rng.integers(0, 2, size=n).tolist())
        if sum(a != b for a, b in zip(row, center)) > radius:
            construction.add(row)
    return sorted(construction), {"strategy": "hamming_ball_and_far_field",
                                  "near_complete": len(near) == ball_size,
                                  "near_count": len(near), "ball_size": ball_size,
                                  "far_count": len(construction) - len(near)}


def sample_holdout(construction, n, budget, rng):
    """Disjoint codes, never a resubstitution test on construction rows."""
    excluded = set(construction)
    remaining = (1 << n) - len(excluded)
    target = min(budget, remaining)
    if target <= 0:
        return []
    if n <= 16 and remaining <= max(1000, 2 * target):
        candidates = [row for row in product((0, 1), repeat=n) if row not in excluded]
        indices = rng.choice(len(candidates), target, replace=False)
        return sorted(candidates[i] for i in indices)
    result = set()
    for _ in range(target * 50):
        if len(result) >= target:
            break
        row = tuple(rng.integers(0, 2, size=n).tolist())
        if row not in excluded:
            result.add(row)
    return sorted(result)
