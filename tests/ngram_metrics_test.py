import numpy as np
import pandas as pd
import pytest
from nltk.translate.bleu_score import SmoothingFunction

from gaico.metrics import BLEU, ROUGE, JSDivergence


# BLEU Tests
class TestBLEU:
    @pytest.fixture(scope="class")
    def bleu_scorer(self):
        return BLEU()

    def test_calculate_simple(self, bleu_scorer, text_pair_simple):
        gen, ref = text_pair_simple
        score = bleu_scorer.calculate(gen, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.254, abs=1e-3)

    def test_calculate_identical(self, bleu_scorer, text_pair_identical):
        gen, ref = text_pair_identical
        score = bleu_scorer.calculate(gen, ref)
        # Will not be 1.0 due to the smoothing function
        assert score == pytest.approx(0.316, abs=1e-3)

    def test_calculate_different(self, bleu_scorer, text_pair_different):
        gen, ref = text_pair_different
        score = bleu_scorer.calculate(gen, ref)
        # Will not be 1.0 due to the smoothing function
        assert score == pytest.approx(0.114, abs=1e-3)

    def test_calculate_empty(self, bleu_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        # TODO
        # NLTK's sentence_bleu might raise ZeroDivisionError or return 0 depending on smoothing
        # Let's expect 0 for simplicity with default smoothing
        score = bleu_scorer.calculate(gen, ref)
        assert score == pytest.approx(0.0)

    def test_batch_calculate_list_corpus(
        self, bleu_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = bleu_scorer.calculate(
            sample_generated_texts, sample_reference_texts, use_corpus_bleu=True
        )
        assert isinstance(scores, float)
        assert 0.0 <= scores <= 1.0

    def test_batch_calculate_list_sentence(
        self, bleu_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = bleu_scorer.calculate(
            sample_generated_texts, sample_reference_texts, use_corpus_bleu=False
        )
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_batch_calculate_np_corpus(
        self, bleu_scorer, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = bleu_scorer.calculate(
            sample_generated_texts_np, sample_reference_texts_np, use_corpus_bleu=True
        )
        assert isinstance(scores, float)

    def test_batch_calculate_np_sentence(
        self,
        bleu_scorer,
        sample_generated_texts_np,
        sample_reference_texts_np,
    ):
        scores = bleu_scorer.calculate(
            sample_generated_texts_np, sample_reference_texts_np, use_corpus_bleu=False
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        assert (
            scores.dtype == np.float64 or scores.dtype == object
        )  # Can be object if mixed types somehow, but expect float
        # Check a specific value
        identical_idx = np.where(sample_generated_texts_np == "identical text")[0][0]
        # Will not be 1.0 due to the smoothing function
        assert scores[identical_idx] == pytest.approx(0.316, abs=1e-3)

    def test_batch_calculate_pd_corpus(
        self, bleu_scorer, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = bleu_scorer.calculate(
            sample_generated_texts_pd, sample_reference_texts_pd, use_corpus_bleu=True
        )
        assert isinstance(scores, float)

    def test_batch_calculate_pd_sentence(
        self,
        bleu_scorer,
        sample_generated_texts_pd,
        sample_reference_texts_pd,
    ):
        scores = bleu_scorer.calculate(
            sample_generated_texts_pd, sample_reference_texts_pd, use_corpus_bleu=False
        )
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        assert scores.dtype == np.float64 or scores.dtype == object
        # Check a specific value
        identical_idx = sample_generated_texts_pd[
            sample_generated_texts_pd == "identical text"
        ].index[0]
        # Will not be 1.0 due to the smoothing function
        assert scores[identical_idx] == pytest.approx(0.316, abs=1e-3)

    def test_bleu_custom_params(self, text_pair_simple):
        bleu_scorer_n2 = BLEU(n=2, smoothing_function=SmoothingFunction().method4)
        gen, ref = text_pair_simple
        score = bleu_scorer_n2.calculate(gen, ref)
        assert isinstance(score, float)
        # Value will differ from default n=4 scorer


# ROUGE Tests
class TestROUGE:
    @pytest.fixture(scope="class")
    def rouge_scorer_default(self):
        return ROUGE()  # Default: ['rouge1', 'rouge2', 'rougeL']

    @pytest.fixture(scope="class")
    def rouge_scorer_l(self):
        return ROUGE(rouge_types=["rougeL"])

    @pytest.fixture(scope="class")
    def rouge_scorer_1_2(self):
        return ROUGE(rouge_types=["rouge1", "rouge2"])

    def test_calculate_default(self, rouge_scorer_default, text_pair_simple):
        gen, ref = text_pair_simple
        score = rouge_scorer_default.calculate(gen, ref)
        assert isinstance(score, dict)
        assert set(score.keys()) == {"rouge1", "rouge2", "rougeL"}
        assert all(isinstance(v, float) for v in score.values())
        assert all(0.0 <= v <= 1.0 for v in score.values())
        assert score["rougeL"] == pytest.approx(5 / 6, abs=1e-3)

    def test_calculate_single_type(self, rouge_scorer_l, text_pair_simple):
        gen, ref = text_pair_simple
        score = rouge_scorer_l.calculate(gen, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(5 / 6, abs=1e-3)

    def test_calculate_multiple_types(self, rouge_scorer_1_2, text_pair_simple):
        gen, ref = text_pair_simple
        score = rouge_scorer_1_2.calculate(gen, ref)
        assert isinstance(score, dict)
        assert set(score.keys()) == {"rouge1", "rouge2"}
        assert all(isinstance(v, float) for v in score.values())

    def test_calculate_identical(self, rouge_scorer_default, text_pair_identical):
        gen, ref = text_pair_identical
        score = rouge_scorer_default.calculate(gen, ref)
        assert score["rouge1"] == pytest.approx(1.0)
        assert score["rouge2"] == pytest.approx(1.0)
        assert score["rougeL"] == pytest.approx(1.0)

    def test_calculate_different(self, rouge_scorer_default, text_pair_different):
        gen, ref = text_pair_different
        score = rouge_scorer_default.calculate(gen, ref)
        assert score["rouge2"] == pytest.approx(0.0)
        assert score["rougeL"] == pytest.approx(1 / 3)
        assert score["rouge1"] == pytest.approx(1 / 3)

    def test_calculate_empty(self, rouge_scorer_default, text_pair_empty):
        gen, ref = text_pair_empty
        score = rouge_scorer_default.calculate(gen, ref)
        # rouge-score typically returns 0 for empty strings
        assert score["rouge1"] == pytest.approx(0.0)
        assert score["rouge2"] == pytest.approx(0.0)
        assert score["rougeL"] == pytest.approx(0.0)

    def test_batch_calculate_list_default(
        self, rouge_scorer_default, sample_generated_texts, sample_reference_texts
    ):
        scores = rouge_scorer_default.calculate(sample_generated_texts, sample_reference_texts)
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, dict) for s in scores)
        assert all(set(s.keys()) == {"rouge1", "rouge2", "rougeL"} for s in scores)

    def test_batch_calculate_list_single(
        self, rouge_scorer_l, sample_generated_texts, sample_reference_texts
    ):
        scores = rouge_scorer_l.calculate(sample_generated_texts, sample_reference_texts)
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, float) for s in scores)

    def test_batch_calculate_np_default(
        self, rouge_scorer_default, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = rouge_scorer_default.calculate(
            sample_generated_texts_np, sample_reference_texts_np
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        assert scores.dtype == object  # Array of dicts

    def test_batch_calculate_np_single(
        self, rouge_scorer_l, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = rouge_scorer_l.calculate(sample_generated_texts_np, sample_reference_texts_np)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        assert scores.dtype == np.float64

    def test_batch_calculate_pd_default(
        self, rouge_scorer_default, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = rouge_scorer_default.calculate(
            sample_generated_texts_pd, sample_reference_texts_pd
        )
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        assert scores.dtype == object  # Series of dicts

    def test_batch_calculate_pd_single(
        self, rouge_scorer_l, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = rouge_scorer_l.calculate(sample_generated_texts_pd, sample_reference_texts_pd)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        assert scores.dtype == np.float64

    def test_rouge_invalid_type_init(self):
        with pytest.raises(ValueError, match="rouge_types must be a list"):
            ROUGE(rouge_types="rougeL")
        with pytest.raises(ValueError, match="rouge_types must be one of"):
            ROUGE(rouge_types=["rouge4"])


# JSDivergence Tests
class TestJSDivergence:
    @pytest.fixture(scope="class")
    def js_divergence_scorer(self):
        try:
            import nltk

            nltk.word_tokenize("test")  # Trigger download if needed
        except LookupError:
            import nltk

            nltk.download("punkt")
        return JSDivergence()

    def test_calculate_simple(self, js_divergence_scorer, text_pair_simple):
        gen, ref = text_pair_simple
        score = js_divergence_scorer.calculate(gen, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # JS Divergence is 0 for identical distributions, 1 for max different.
        # Score is 1 - JSD, so 1 for identical, 0 for max different.
        assert score == pytest.approx(0.660, abs=1e-3)

    def test_calculate_identical(self, js_divergence_scorer, text_pair_identical):
        gen, ref = text_pair_identical
        score = js_divergence_scorer.calculate(gen, ref)
        assert score == pytest.approx(1.0)

    def test_calculate_different(self, js_divergence_scorer, text_pair_different):
        gen, ref = text_pair_different
        score = js_divergence_scorer.calculate(gen, ref)
        assert score == pytest.approx(0.320, abs=1e-3)

    def test_calculate_empty(self, js_divergence_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        score = js_divergence_scorer.calculate(gen, ref)
        # Implementation returns 0.0 if either distribution is all zeros
        assert score == pytest.approx(0.0)

    def test_calculate_one_empty(self, js_divergence_scorer, text_pair_one_empty):
        gen, ref = text_pair_one_empty
        score = js_divergence_scorer.calculate(gen, ref)
        # Implementation returns 0.0 if either distribution is all zeros
        assert score == pytest.approx(0.0)

    def test_batch_calculate_list(
        self, js_divergence_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = js_divergence_scorer.calculate(sample_generated_texts, sample_reference_texts)
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_batch_calculate_np(
        self, js_divergence_scorer, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = js_divergence_scorer.calculate(
            sample_generated_texts_np, sample_reference_texts_np
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        assert scores.dtype == np.float64

    def test_batch_calculate_pd(
        self, js_divergence_scorer, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = js_divergence_scorer.calculate(
            sample_generated_texts_pd, sample_reference_texts_pd
        )
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        assert scores.dtype == np.float64
