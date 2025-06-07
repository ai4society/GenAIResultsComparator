import numpy as np
import pandas as pd
import pytest

# Conditional import for SmoothingFunction for type hinting if NLTK is available
try:
    from nltk.translate.bleu_score import SmoothingFunction

    _nltk_smoothing_function_available = True
except ImportError:
    SmoothingFunction = None  # type: ignore
    _nltk_smoothing_function_available = False


from gaico.metrics import BLEU, ROUGE, JSDivergence

# Import flags to check for dependency availability
from gaico.metrics.ngram_metrics import __nltk_available__, __scipy_available__


# BLEU Tests
@pytest.mark.skipif(not __nltk_available__, reason="NLTK not installed, skipping BLEU tests.")
class TestBLEU:
    @pytest.fixture(scope="class")
    def bleu_scorer(self):
        # This will raise ImportError if NLTK is not available,
        # but skipif should prevent this fixture from running.
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
        assert score == pytest.approx(0.316, abs=1e-3)  # Default smoothing method1

    def test_calculate_different(self, bleu_scorer, text_pair_different):
        gen, ref = text_pair_different
        score = bleu_scorer.calculate(gen, ref)
        # Will not be 1.0 due to the smoothing function
        assert score == pytest.approx(0.114, abs=1e-3)  # Default smoothing method1

    def test_calculate_empty(self, bleu_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        # NLTK's sentence_bleu might raise ZeroDivisionError or return 0 depending on smoothing
        # With default smoothing (method1), it should return 0.0 for empty strings.
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
        # NLTK BLEU returns float64
        assert scores.dtype == np.float64
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
        assert scores.dtype == np.float64
        # Check a specific value
        identical_idx = sample_generated_texts_pd[
            sample_generated_texts_pd == "identical text"
        ].index[0]
        # Will not be 1.0 due to the smoothing function
        assert scores[identical_idx] == pytest.approx(0.316, abs=1e-3)  # type: ignore

    def test_bleu_custom_params(self, text_pair_simple):
        # This test implicitly checks if SmoothingFunction can be imported and used
        if not _nltk_smoothing_function_available:
            pytest.skip("NLTK SmoothingFunction not available for custom BLEU params test.")

        # Ensure SmoothingFunction is not None before calling its methods
        assert SmoothingFunction is not None, (
            "SmoothingFunction should be imported if NLTK is available"
        )

        bleu_scorer_n2 = BLEU(n=2, smoothing_function=SmoothingFunction().method4)  # type: ignore
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
        assert scores.dtype == np.float64  # ROUGE with single type returns float

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
        assert scores.dtype == np.float64  # ROUGE with single type returns float

    def test_rouge_invalid_type_init(self):
        with pytest.raises(ValueError, match="rouge_types must be a list"):
            ROUGE(rouge_types="rougeL")  # type: ignore
        with pytest.raises(ValueError, match="rouge_types must be one of"):
            ROUGE(rouge_types=["rouge4"])


# JSDivergence Tests
@pytest.mark.skipif(
    not (__nltk_available__ and __scipy_available__),
    reason="NLTK and/or SciPy not installed, skipping JSDivergence tests.",
)
class TestJSDivergence:
    @pytest.fixture(scope="class")
    def js_divergence_scorer(self):
        # This fixture will only run if the skipif condition is false.
        # The JSDivergence() constructor will raise ImportError if deps are missing,
        # but skipif should prevent that.
        try:
            import nltk

            # Attempt to use a tokenizer to trigger NLTK data download if not present
            # This is a common point of failure if 'punkt' is missing.
            nltk.word_tokenize("test download trigger")
        except LookupError:
            # If 'punkt' is missing, download it.
            # This is best effort; ideally, users manage NLTK data.
            # In a CI environment, NLTK data should be pre-cached.
            print("\nNLTK 'punkt' resource not found. Attempting download...")
            try:
                nltk.download("punkt", quiet=True)  # type: ignore
                print("NLTK 'punkt' downloaded successfully.")
            except Exception as e:
                print(f"Failed to download NLTK 'punkt': {e}. JSD tests might be unreliable.")
        except ImportError:
            # NLTK itself is not available, skipif should have caught this.
            pass
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
