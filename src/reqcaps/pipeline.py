"""Construct once per audited instance, reuse for all four query families."""
from dataclasses import dataclass
import numpy as np
from .config import Config
from .mapping import BlackBox, LocalMapping, build_mapping, sample_construction, sample_holdout, validate_domain
from .bdd import ROBDD
from .automaton import Automaton
from .queries import QueryRuntime


@dataclass
class AuditArtifact:
    target_label: object
    mapping: LocalMapping
    circuit: ROBDD
    automaton: Automaton
    runtime: QueryRuntime
    audit: dict
    config: Config

    def answer_all(self, costs=None):
        return self.runtime.answer_all(costs=costs)

    def to_dict(self, *, include_structure=True, costs=None):
        label = self.target_label.item() if hasattr(self.target_label, "item") else self.target_label
        value = {"schema_version": 1, "target_label": label,
                 "scope": "finite_processed_feature_reconstruction",
                 "compiled_semantics": "positive_construction_minterms_closed_world",
                 "mapping": self.mapping.to_dict(), "audit": self.audit,
                 "config": self.config.to_dict(), "answers": self.answer_all(costs),
                 "sharing": self.runtime.stats()}
        if include_structure:
            value["circuit"] = self.circuit.to_dict()
            value["automaton"] = self.automaton.to_dict()
        return value


class ReQCAPS:
    def __init__(self, config=None):
        self.config = config or Config()

    def build(self, model, instance, background, *, validity=None, feature_names=None):
        """Build a fresh frozen object. `validity(X)` optionally checks processed vectors.

        A failed fidelity/center check remains visible in audit flags. The returned
        circuit and DFA are diagnostic objects; flagged answers are not certified
        explanations of the original black-box decision.
        """
        config = self.config
        rng = np.random.default_rng(config.seed)
        blackbox = BlackBox(model)
        mapping, target = build_mapping(blackbox, instance, background, config, rng, validity)
        if feature_names is not None and len(feature_names) != len(mapping.anchor):
            raise ValueError("feature_names must match the processed input dimension")
        screening_queries = blackbox.query_count
        center = mapping.encode(mapping.anchor)
        sampled, sampling_metadata = sample_construction(center, config, rng)
        construction = sampled
        held = []
        if config.holdout_fraction and len(sampled) > 1:
            candidates = [row for row in sampled if row != center]
            count = min(len(candidates), max(1, int(len(sampled) * config.holdout_fraction)))
            held_indices = rng.choice(len(candidates), count, replace=False)
            held = sorted(candidates[i] for i in held_indices)
            held_set = set(held)
            construction = [row for row in sampled if row not in held_set]
        # In exact mode the full cube is construction: no disjoint hold-out exists.
        extra = sample_holdout(sampled, mapping.n, max(0, config.holdout_budget - len(held)), rng)
        held += extra

        def label(rows):
            if not rows:
                return []
            reconstructed = mapping.reconstruct_batch(rows)
            validate_domain(validity, reconstructed, "reconstruction")
            return (blackbox.predict(reconstructed) == target).astype(int).tolist()

        construction_labels = label(construction)
        held_labels = label(held)
        order = tuple(range(mapping.n))
        if config.ordering in {"importance", "sift"}:
            order = tuple(sorted(range(mapping.n), key=lambda i: (-mapping.importance[mapping.selected[i]],
                                                                 mapping.selected[i])))
        diagram = ROBDD.compile(construction, construction_labels, n=mapping.n, order=order,
                                node_budget=config.node_budget,
                                sifting_passes=config.sifting_passes if config.ordering == "sift" else 0)
        automaton = Automaton(diagram)
        evidence = dict(zip(construction, construction_labels))
        evidence.update(zip(held, held_labels))
        runtime = QueryRuntime(diagram, automaton, center, feature_indices=mapping.selected,
                               subset_budget=config.why_subset_budget,
                               raw_evidence=evidence)
        constructed_matches = [diagram.evaluate(row) == value for row, value in zip(construction, construction_labels)]
        holdout_fidelity = (float(np.mean([diagram.evaluate(row) == value for row, value in zip(held, held_labels)]))
                            if held else None)
        bridge_rows = construction + held
        agreement = float(np.mean([automaton.run(row) == diagram.evaluate(row) for row in bridge_rows]))
        reconstructed_center_value = evidence[center]
        flags = []
        if not reconstructed_center_value:
            flags.append("center_reconstruction_disagreement")
        if holdout_fidelity is not None and holdout_fidelity < config.fidelity_threshold:
            flags.append("holdout_fidelity_below_threshold")
        if not diagram.complete and holdout_fidelity is None:
            flags.append("incomplete_construction_without_holdout")
        if agreement != 1.0:
            flags.append("circuit_automaton_disagreement")
        audit = {"flags": flags, "requires_review": bool(flags),
                 "center_reconstruction_preserves_label": bool(reconstructed_center_value),
                 "center_compiled_output": diagram.evaluate(center),
                 "construction_count": len(construction), "holdout_count": len(held),
                 "construction_complete": diagram.complete,
                 "known_raw_codes": len(evidence), "raw_domain_complete": runtime.raw_complete,
                 "construction_agreement": float(np.mean(constructed_matches)),
                 "holdout_fidelity": holdout_fidelity,
                 "holdout_interpretation": ("disjoint_reconstructed_codes" if held else
                                            "not_available_full_cube_constructed" if diagram.complete else
                                            "not_available_incomplete_construction"),
                 "circuit_automaton_agreement_on_observed_codes": agreement,
                 "blackbox_rows_evaluated": blackbox.query_count,
                 "screening_rows_evaluated_including_anchor": screening_queries,
                 "reconstruction_rows_evaluated": blackbox.query_count - screening_queries,
                 "sampling": sampling_metadata,
                 "raw_domain_actionability_certified": False}
        if feature_names is not None:
            audit["selected_feature_names"] = [feature_names[i] for i in mapping.selected]
        return AuditArtifact(target, mapping, diagram, automaton, runtime, audit, config)

    def build_batch(self, model, instances, background, **kwargs):
        """One object per instance; no unjustified circuit sharing across anchors."""
        return [self.build(model, instance, background, **kwargs) for instance in instances]

    def answer_batch(self, artifacts, requests, *, costs=None):
        """Reuse built objects across arbitrary (instance index, family) requests.

        Construction, black-box calls, compilation and bridge do not recur here.
        `costs` is a mapping from instance index to local cost vector, if supplied.
        """
        result = []
        for instance_index, family in requests:
            vector = None if costs is None else costs.get(instance_index)
            result.append(artifacts[instance_index].runtime.answer(family, costs=vector))
        return result

