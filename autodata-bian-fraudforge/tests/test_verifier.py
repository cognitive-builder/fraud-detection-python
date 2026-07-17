from fraudforge.config import Recipe
from fraudforge.generator import SyntheticTransactionGenerator
from fraudforge.verifier import DeterministicVerifier


def test_verifier_detects_label_corruption() -> None:
    frame = SyntheticTransactionGenerator(Recipe(), seed=22).generate(350)
    frame.loc[0, "oracle_label"] = 1 - int(frame.loc[0, "oracle_label"])

    result = DeterministicVerifier().verify(frame)

    assert not bool(result.rows.loc[0, "verifier_pass"])
    assert result.rows.loc[0, "verifier_reason"] == "LABEL_SCENARIO_MISMATCH"
    assert 0.95 <= result.validity_rate < 1.0
    assert result.family_coverage == 1.0
