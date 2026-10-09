from itertools import product
import json
import numpy as np
import pytest
from reqcaps import ReQCAPS, Config
from reqcaps.mapping import sample_construction, sample_holdout
from reqcaps.cli import main


def and_classifier(values):
    return np.where((values[:, 0] > 0) & (values[:, 1] > 0), "yes", "no")


def background(n=2):
    return np.asarray(list(product((-1., 1.), repeat=n)))


def test_pipeline_full_construction_roundtrip_and_no_rebuild():
    calls = []
    def predictor(values):
        calls.append(len(values))
        return and_classifier(values)
    engine = ReQCAPS(Config(feature_budget=2, background_samples=4))
    artifact = engine.build(predictor, np.ones(2), background())
    assert artifact.target_label == "yes"
    assert artifact.audit["holdout_fidelity"] is None
    assert artifact.audit["construction_complete"]
    assert artifact.audit["holdout_interpretation"] == "not_available_full_cube_constructed"
    assert not artifact.audit["flags"]
    before = list(calls)
    answers = engine.answer_batch([artifact], [(0, "why"), (0, "why_not"),
                                               (0, "interaction"), (0, "sensitivity")] * 2)
    assert calls == before
    assert answers[0]["reasons"] == [[0, 1]]
    assert artifact.answer_all()["sensitivity"]["minimum_hamming_distance"] == 1
    assert artifact.runtime.stats()["cached_family_answers"] == 4
    json.dumps(artifact.to_dict(), allow_nan=False)
    for row in product((0, 1), repeat=2):
        assert artifact.mapping.encode(artifact.mapping.reconstruct(row)) == row


def test_split_is_disjoint_keeps_center_and_reports_actual_holdout():
    model = lambda values: (np.sum(values, axis=1) > -4).astype(int)
    config = Config(feature_budget=3, holdout_fraction=.2)
    artifact = ReQCAPS(config).build(model, np.ones(3), background(3))
    assert not artifact.circuit.complete
    assert artifact.mapping.encode(np.ones(3)) in artifact.circuit.observations
    assert artifact.audit["holdout_count"] == 1
    assert artifact.audit["holdout_fidelity"] == 0
    assert artifact.audit["holdout_interpretation"] == "disjoint_reconstructed_codes"
    assert "holdout_fidelity_below_threshold" in artifact.audit["flags"]
    assert artifact.runtime.raw_complete
    assert not artifact.runtime.raw_equivalent


def test_center_lossy_reconstruction_is_flagged():
    model = lambda values: (values[:, 0] > 5).astype(int)
    artifact = ReQCAPS(Config(feature_budget=1)).build(model, [9.], [[0.], [1.], [2.], [3.]])
    assert "center_reconstruction_disagreement" in artifact.audit["flags"]
    assert artifact.audit["center_compiled_output"] == 0


def test_degenerate_feature_dropped_and_zero_dimensional_artifact():
    model = lambda values: np.ones(len(values), dtype=int)
    artifact = ReQCAPS().build(model, [3., 3.], [[3., 3.], [3., 3.]])
    assert artifact.mapping.n == 0
    assert artifact.mapping.dropped == (0, 1)
    assert artifact.answer_all()["why"]["reasons"] == [[]]
    assert artifact.audit["construction_complete"]


def test_sampling_and_holdout_are_replayable_and_disjoint():
    config = Config(feature_budget=6, enumeration_limit=2, hamming_radius=1,
                    near_budget=10, far_budget=6, holdout_budget=5)
    one, metadata = sample_construction((1,) * 6, config, np.random.default_rng(42))
    two, _ = sample_construction((1,) * 6, config, np.random.default_rng(42))
    assert one == two and metadata["near_complete"]
    held = sample_holdout(one, 6, 5, np.random.default_rng(99))
    assert len(held) == 5 and not (set(one) & set(held))
    artifact = ReQCAPS(config).build(lambda values: (values.sum(axis=1) > 0).astype(int),
                                   np.ones(6), background(6))
    assert artifact.audit["holdout_count"] == 5
    assert not artifact.audit["construction_complete"]


def test_incomplete_construction_without_holdout_has_accurate_metadata():
    config = Config(feature_budget=2, enumeration_limit=0, hamming_radius=0,
                    near_budget=1, far_budget=0, holdout_budget=0)
    artifact = ReQCAPS(config).build(and_classifier, np.ones(2), background())
    assert artifact.audit["construction_count"] == 1
    assert not artifact.audit["construction_complete"]
    assert artifact.audit["holdout_count"] == 0
    assert artifact.audit["holdout_fidelity"] is None
    assert artifact.audit["holdout_interpretation"] == "not_available_incomplete_construction"
    assert "incomplete_construction_without_holdout" in artifact.audit["flags"]


def test_validity_predicate_is_not_silently_ignored():
    with pytest.raises(ValueError, match="validity"):
        ReQCAPS(Config(feature_budget=1)).build(lambda values: np.zeros(len(values)),
                                              [9.], [[0.], [2.], [4.], [6.]],
                                              validity=lambda values: values[:, 0] >= 8)


@pytest.mark.parametrize("fields", [{"feature_budget": 0}, {"enumeration_limit": 30},
                                    {"holdout_fraction": 1}, {"ordering": "random"}])
def test_invalid_config(fields):
    with pytest.raises(ValueError):
        Config(**fields)


def test_cli_is_quiet_and_writes_no_result_by_default(tmp_path, monkeypatch, capsys):
    import sys
    import types
    module = types.ModuleType("toy_predictor")
    module.predict = and_classifier
    monkeypatch.setitem(sys.modules, module.__name__, module)
    background_path, instances_path = tmp_path / "background.csv", tmp_path / "instances.csv"
    np.savetxt(background_path, background(), delimiter=",")
    np.savetxt(instances_path, np.ones((1, 2)), delimiter=",")
    before = set(tmp_path.iterdir())
    assert main(["--model", "toy_predictor:predict", "--background", str(background_path),
                 "--instances", str(instances_path)]) == 0
    assert set(tmp_path.iterdir()) == before
    assert capsys.readouterr().out == ""
