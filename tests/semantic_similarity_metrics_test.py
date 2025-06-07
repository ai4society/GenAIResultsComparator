import numpy as np
import pandas as pd
import pytest

from gaico.metrics import BERTScore

# Import the flag to check for dependency availability
from gaico.metrics.semantic_similarity_metrics import __semantic_deps_available__


# This test checks GAICo's BERTScore wrapper's __init__ validation.
# It should run even if bert-score/torch are not installed,
# as the ValueError should be raised before the ImportError for missing deps.
def test_bertscore_invalid_output_val_init_standalone():
    with pytest.raises(ValueError, match="`output_val` must be a list"):
        BERTScore(output_val="f1")  # type: ignore
    with pytest.raises(ValueError, match="`output_val` must be one of"):
        BERTScore(output_val=["f2"])  # type: ignore


# Mark tests as potentially slow due to model download/loading
@pytest.mark.bertscore
class TestBERTScore:
    # Skip all tests in this class if bert-score or its dependencies are not installed
    pytestmark = pytest.mark.skipif(
        not __semantic_deps_available__,
        reason="bert-score and/or its dependencies (torch) not installed, skipping BERTScore calculation tests.",
    )

    # Use class scope to load model only once per test class run
    @pytest.fixture(scope="class")
    def bert_scorer_default(self):
        # This might download the model on first run
        return BERTScore()

    @pytest.fixture(scope="class")
    def bert_scorer_f1(self):
        return BERTScore(output_val=["f1"])

    @pytest.fixture(scope="class")
    def bert_scorer_pr(self):
        return BERTScore(output_val=["precision", "recall"])

    def test_calculate_default(self, bert_scorer_default, text_pair_simple):
        gen, ref = text_pair_simple
        score = bert_scorer_default.calculate(gen, ref)
        assert isinstance(score, dict)
        assert set(score.keys()) == {"precision", "recall", "f1"}
        assert all(isinstance(v, float) for v in score.values())
        # BERTScore values are typically high for similar sentences
        assert all(0.75 <= v <= 1.0 for v in score.values())  # Rough check

    def test_calculate_single_type(self, bert_scorer_f1, text_pair_simple):
        gen, ref = text_pair_simple
        score = bert_scorer_f1.calculate(gen, ref)
        assert isinstance(score, float)
        assert 0.75 <= score <= 1.0  # Rough check

    def test_calculate_multiple_types(self, bert_scorer_pr, text_pair_simple):
        gen, ref = text_pair_simple
        score = bert_scorer_pr.calculate(gen, ref)
        assert isinstance(score, dict)
        assert set(score.keys()) == {"precision", "recall"}
        assert all(isinstance(v, float) for v in score.values())

    def test_calculate_identical(self, bert_scorer_default, text_pair_identical):
        gen, ref = text_pair_identical
        score = bert_scorer_default.calculate(gen, ref)
        assert score["precision"] == pytest.approx(1.0, abs=1e-5)
        assert score["recall"] == pytest.approx(1.0, abs=1e-5)
        assert score["f1"] == pytest.approx(1.0, abs=1e-5)

    def test_calculate_different(self, bert_scorer_default, text_pair_different):
        gen, ref = text_pair_different
        score = bert_scorer_default.calculate(gen, ref)
        assert score["f1"] < 0.75

    def test_calculate_empty(self, bert_scorer_default, text_pair_empty):
        gen, ref = text_pair_empty
        # BERTScore might produce NaNs or errors with empty strings, or low scores
        # Let's check it doesn't crash and returns low scores/zeros
        try:
            score = bert_scorer_default.calculate(gen, ref)
            assert score["precision"] == pytest.approx(0.0, abs=1e-5)
            assert score["recall"] == pytest.approx(0.0, abs=1e-5)
            assert score["f1"] == pytest.approx(0.0, abs=1e-5)
        except Exception as e:
            # TODO: Allow specific expected exceptions if the library handles them gracefully
            pytest.fail(f"BERTScore calculation failed on empty strings: {e}")

    def test_batch_calculate_list_default(
        self, bert_scorer_default, sample_generated_texts, sample_reference_texts
    ):
        # Filter out empty strings if they cause issues, or handle expected output
        gen_filtered = [t for t in sample_generated_texts if t]
        ref_filtered = [
            sample_reference_texts[i] for i, t in enumerate(sample_generated_texts) if t
        ]

        if not gen_filtered:  # Skip if only empty strings remain
            pytest.skip("Skipping batch test with only empty strings")

        scores = bert_scorer_default.calculate(gen_filtered, ref_filtered)
        assert isinstance(scores, list)
        assert len(scores) == len(gen_filtered)
        assert all(isinstance(s, dict) for s in scores)
        assert all(set(s.keys()) == {"precision", "recall", "f1"} for s in scores)

    def test_batch_calculate_list_single(
        self, bert_scorer_f1, sample_generated_texts, sample_reference_texts
    ):
        gen_filtered = [t for t in sample_generated_texts if t]
        ref_filtered = [
            sample_reference_texts[i] for i, t in enumerate(sample_generated_texts) if t
        ]

        if not gen_filtered:
            pytest.skip("Skipping batch test with only empty strings")

        scores = bert_scorer_f1.calculate(gen_filtered, ref_filtered)
        assert isinstance(scores, list)
        assert len(scores) == len(gen_filtered)
        assert all(
            isinstance(score, float) for score_dict in scores for score in score_dict.values()
        )

    def test_batch_calculate_np_default(
        self, bert_scorer_default, sample_generated_texts_np, sample_reference_texts_np
    ):
        mask = sample_generated_texts_np != ""
        gen_filtered = sample_generated_texts_np[mask]
        ref_filtered = sample_reference_texts_np[mask]

        if gen_filtered.size == 0:
            pytest.skip("Skipping batch test with only empty strings")

        scores = bert_scorer_default.calculate(gen_filtered, ref_filtered)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(gen_filtered)
        assert scores.dtype == object  # Array of dicts

    def test_batch_calculate_np_single(
        self, bert_scorer_f1, sample_generated_texts_np, sample_reference_texts_np
    ):
        mask = sample_generated_texts_np != ""
        gen_filtered = sample_generated_texts_np[mask]
        ref_filtered = sample_reference_texts_np[mask]

        if gen_filtered.size == 0:
            pytest.skip("Skipping batch test with only empty strings")

        scores = bert_scorer_f1.calculate(gen_filtered, ref_filtered)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(gen_filtered)
        assert all(
            isinstance(score, float) for score_dict in scores for score in score_dict.values()
        )

    def test_batch_calculate_pd_default(
        self, bert_scorer_default, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        mask = sample_generated_texts_pd != ""
        gen_filtered = sample_generated_texts_pd[mask]
        ref_filtered = sample_reference_texts_pd[mask]

        if gen_filtered.empty:
            pytest.skip("Skipping batch test with only empty strings")

        scores = bert_scorer_default.calculate(gen_filtered, ref_filtered)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(gen_filtered)
        assert scores.dtype == object  # Series of dicts

    def test_batch_calculate_pd_single(
        self, bert_scorer_f1, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        mask = sample_generated_texts_pd != ""
        gen_filtered = sample_generated_texts_pd[mask]
        ref_filtered = sample_reference_texts_pd[mask]

        if gen_filtered.empty:
            pytest.skip("Skipping batch test with only empty strings")

        scores = bert_scorer_f1.calculate(gen_filtered, ref_filtered)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(gen_filtered)
        assert all(
            isinstance(score, float) for score_dict in scores for score in score_dict.values()
        )
