"""Explicit, serializable construction and query budgets."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    feature_budget: int = 10
    background_samples: int = 50
    enumeration_limit: int = 15
    hamming_radius: int = 3
    near_budget: int = 2000
    far_budget: int = 200
    holdout_budget: int = 200
    holdout_fraction: float = 0.0
    seed: int = 42
    ordering: str = "sift"
    sifting_passes: int = 2
    node_budget: int = 50000
    why_subset_budget: int = 100000
    fidelity_threshold: float = 0.9

    def __post_init__(self):
        for name in ("feature_budget", "background_samples", "near_budget",
                     "node_budget", "why_subset_budget"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("enumeration_limit", "hamming_radius", "far_budget",
                     "holdout_budget", "sifting_passes", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.enumeration_limit > 20:
            raise ValueError("enumeration_limit must be <=20 to bound allocation")
        if self.feature_budget > 64:
            raise ValueError("feature_budget must be <=64")
        if self.node_budget < 2:
            raise ValueError("node_budget must allow both terminals")
        if self.ordering not in {"natural", "importance", "sift"}:
            raise ValueError("ordering must be natural, importance, or sift")
        if not 0 <= self.fidelity_threshold <= 1:
            raise ValueError("fidelity_threshold must lie in [0,1]")
        if not 0 <= self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must lie in [0,1)")

    def to_dict(self):
        return asdict(self)

