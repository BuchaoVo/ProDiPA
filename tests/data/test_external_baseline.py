from scripts.run_external_baseline import make_offspring, select_population


def test_nsga_population_and_mutation_are_bounded():
    rows = [
        {"sequence_id": "a", "sequence": "A" * 50, "fold_plddt_100": "90", "solubility_scaled_01": ".5", "thermophilicity_score_01": ".5"},
        {"sequence_id": "b", "sequence": "C" * 50, "fold_plddt_100": "80", "solubility_scaled_01": ".8", "thermophilicity_score_01": ".6"},
        {"sequence_id": "c", "sequence": "D" * 50, "fold_plddt_100": "50", "solubility_scaled_01": ".4", "thermophilicity_score_01": ".3"},
    ]
    selected = select_population(rows, 2)
    assert {row["sequence_id"] for row in selected} == {"a", "b"}
    children = make_offspring(selected, generation=0, size=4, seed=42)
    assert len(children) == len({row["sequence"] for row in children}) == 4
    assert all(len(row["sequence"]) == 50 for row in children)
