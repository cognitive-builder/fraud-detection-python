from fraudforge.config import FRAUD_SCENARIOS, Recipe
from fraudforge.generator import SyntheticTransactionGenerator, evidence_flags


def test_generator_is_reproducible_and_covers_schema() -> None:
    recipe = Recipe(fraud_prevalence=0.45)
    first = SyntheticTransactionGenerator(recipe, seed=123).generate(500)
    second = SyntheticTransactionGenerator(recipe, seed=123).generate(500)

    assert first.equals(second)
    assert len(first) == 500
    assert set(FRAUD_SCENARIOS).issubset(set(first["scenario_type"]))
    assert first["transaction_id"].is_unique
    assert first["amount"].gt(0).all()
    assert first["oracle_label"].isin([0, 1]).all()


def test_planted_fraud_has_detectable_family_evidence() -> None:
    frame = SyntheticTransactionGenerator(
        Recipe(fraud_prevalence=0.50, ambiguous_rate=0.0, noise=0.0), seed=77
    ).generate(600)
    flags = evidence_flags(frame)
    fraud = frame[frame["oracle_label"] == 1]
    passed = [bool(flags.loc[index, row["scenario_type"]]) for index, row in fraud.iterrows()]
    assert sum(passed) / len(passed) >= 0.99
