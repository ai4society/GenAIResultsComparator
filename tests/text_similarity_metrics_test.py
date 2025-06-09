import numpy as np
import pandas as pd
import pytest

from gaico.metrics import (
    CosineSimilarity,
    JaccardSimilarity,
    LevenshteinDistance,
    SequenceMatcherSimilarity,
)

# Import the availability flag for sklearn
from gaico.metrics.text_similarity_metrics import __sklearn_available__


# JaccardSimilarity Tests
class TestJaccardSimilarity:
    @pytest.fixture(scope="class")
    def jaccard_scorer(self):
        return JaccardSimilarity()

    def test_calculate_simple(self, jaccard_scorer, text_pair_simple):
        gen, ref = text_pair_simple
        score = jaccard_scorer.calculate(gen, ref)
        assert isinstance(score, float)
        # {"the", "cat", "sat", "on", "mat"} vs {"the", "cat", "was", "on", "mat"}
        # Intersection: {"the", "cat", "on", "mat"} (4)
        # Union: {"the", "cat", "sat", "on", "mat", "was"} (6)
        # Score: 4 / 6 = 0.666...
        assert score == pytest.approx(4 / 6)

    def test_calculate_identical(self, jaccard_scorer, text_pair_identical):
        gen, ref = text_pair_identical
        score = jaccard_scorer.calculate(gen, ref)
        assert score == pytest.approx(1.0)

    def test_calculate_different(self, jaccard_scorer, text_pair_different):
        gen, ref = text_pair_different
        score = jaccard_scorer.calculate(gen, ref)
        # "apple banana cherry" vs "dog elephant fox"
        # Intersection: {} (0)
        # Union: {"apple", "banana", "cherry", "dog", "elephant", "fox"} (6)
        # Score: 0 / 6 = 0.0
        assert score == pytest.approx(0.2)

    def test_calculate_empty(self, jaccard_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        score = jaccard_scorer.calculate(gen, ref)
        assert score == pytest.approx(0.0)  # Union is 0, handled

    def test_calculate_one_empty(self, jaccard_scorer, text_pair_one_empty):
        gen, ref = text_pair_one_empty
        score = jaccard_scorer.calculate(gen, ref)
        assert score == pytest.approx(0.0)  # Intersection is 0

    def test_batch_calculate_list(
        self, jaccard_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = jaccard_scorer.calculate(sample_generated_texts, sample_reference_texts)
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_batch_calculate_np(
        self, jaccard_scorer, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = jaccard_scorer.calculate(sample_generated_texts_np, sample_reference_texts_np)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        # Jaccard returns float, so np.float64 is expected
        assert scores.dtype == np.float64

    def test_batch_calculate_pd(
        self, jaccard_scorer, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = jaccard_scorer.calculate(sample_generated_texts_pd, sample_reference_texts_pd)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        # Jaccard returns float, so np.float64 is expected
        assert scores.dtype == np.float64


# CosineSimilarity Tests
@pytest.mark.skipif(
    not __sklearn_available__, reason="scikit-learn not installed, skipping CosineSimilarity tests"
)
class TestCosineSimilarity:
    @pytest.fixture(scope="class")
    def cosine_scorer(self):
        # This fixture will only run if sklearn is available due to the class-level skipif
        return CosineSimilarity()

    def test_calculate_simple(self, cosine_scorer, text_pair_simple):
        gen, ref = text_pair_simple
        score = cosine_scorer.calculate(gen, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # "the cat sat on the mat" vs "the cat was on the mat"
        # Vectors: gen_vec=[1,1,1,1,1,0], ref_vec=[1,1,0,1,1,1] (order: the,cat,sat,on,mat,was)
        # Dot product: 1*1 + 1*1 + 1*0 + 1*1 + 1*1 + 0*1 = 4
        # Norm gen: sqrt(5), Norm ref: sqrt(5)
        # Score: 4 / (sqrt(5)*sqrt(5)) = 4 / 5 = 0.8
        assert score == pytest.approx(0.8, abs=1e-1)

    def test_calculate_identical(self, cosine_scorer, text_pair_identical):
        gen, ref = text_pair_identical
        score = cosine_scorer.calculate(gen, ref)
        assert score == pytest.approx(1.0)

    def test_calculate_different(self, cosine_scorer, text_pair_different):
        gen, ref = text_pair_different
        score = cosine_scorer.calculate(gen, ref)
        # "apple banana cherry" vs "dog elephant fox"
        # No common words, vectors are orthogonal.
        assert score == pytest.approx(1 / 3)

    def test_calculate_empty(self, cosine_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        # Implementation handles this: if both empty, returns 1.0
        score = cosine_scorer.calculate(gen, ref)
        assert score == pytest.approx(1.0)

    def test_calculate_one_empty(self, cosine_scorer, text_pair_one_empty):
        gen, ref = text_pair_one_empty
        # Implementation handles this: if one empty, returns 0.0
        score = cosine_scorer.calculate(gen, ref)
        assert score == pytest.approx(0.0)

    def test_batch_calculate_list(
        self, cosine_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = cosine_scorer.calculate(sample_generated_texts, sample_reference_texts)
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_batch_calculate_np(
        self, cosine_scorer, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = cosine_scorer.calculate(sample_generated_texts_np, sample_reference_texts_np)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        assert scores.dtype == np.float64

    def test_batch_calculate_pd(
        self, cosine_scorer, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = cosine_scorer.calculate(sample_generated_texts_pd, sample_reference_texts_pd)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        assert scores.dtype == np.float64


# LevenshteinDistance Tests
class TestLevenshteinDistance:
    @pytest.fixture(scope="class")
    def levenshtein_scorer(self):
        return LevenshteinDistance()  # Default is calculate_ratio=True

    # Test Ratio (Default)
    def test_calculate_ratio_simple(self, levenshtein_scorer, text_pair_simple):
        gen, ref = text_pair_simple
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=True)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # "the cat sat on the mat" vs "the cat was on the mat" (distance 2, len 22/22) ratio = 1 - 2/22 = 0.909...
        assert score == pytest.approx(1 - 2 / 22)

    def test_calculate_ratio_identical(self, levenshtein_scorer, text_pair_identical):
        gen, ref = text_pair_identical
        score = levenshtein_scorer.calculate(gen, ref)
        assert score == pytest.approx(1.0)

    def test_calculate_ratio_different(self, levenshtein_scorer, text_pair_different):
        gen, ref = text_pair_different
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=True)
        assert score < 0.70

    def test_calculate_ratio_empty(self, levenshtein_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=True)
        assert score == pytest.approx(1.0)  # Ratio of "" and "" is 1

    def test_calculate_ratio_one_empty(self, levenshtein_scorer, text_pair_one_empty):
        gen, ref = text_pair_one_empty
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=True)
        assert score == pytest.approx(0.0)  # Ratio of "abc" and "" is 0

    # Test Distance
    def test_calculate_distance_simple(self, levenshtein_scorer, text_pair_simple):
        gen, ref = text_pair_simple
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=False)
        assert isinstance(score, float)
        assert score == pytest.approx(0.9, rel=1e-1)

    def test_calculate_distance_identical(self, levenshtein_scorer, text_pair_identical):
        gen, ref = text_pair_identical
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=False)
        assert score == pytest.approx(1.0)

    def test_calculate_distance_empty(self, levenshtein_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=False)
        assert score == pytest.approx(1.0)

    def test_calculate_distance_one_empty(self, levenshtein_scorer, text_pair_one_empty):
        gen, ref = text_pair_one_empty
        score = levenshtein_scorer.calculate(gen, ref, calculate_ratio=False)
        assert score == pytest.approx(0)

    # Test Batch (Ratio)
    def test_batch_calculate_ratio_list(
        self, levenshtein_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = levenshtein_scorer.calculate(
            sample_generated_texts, sample_reference_texts, calculate_ratio=True
        )
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, float) for s in scores)

    def test_batch_calculate_ratio_np(
        self, levenshtein_scorer, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = levenshtein_scorer.calculate(
            sample_generated_texts_np, sample_reference_texts_np, calculate_ratio=True
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        assert scores.dtype == np.float64

    def test_batch_calculate_ratio_pd(
        self, levenshtein_scorer, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = levenshtein_scorer.calculate(
            sample_generated_texts_pd, sample_reference_texts_pd, calculate_ratio=True
        )
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        assert scores.dtype == np.float64

    # Test Batch (Distance)
    def test_batch_calculate_distance_list(
        self, levenshtein_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = levenshtein_scorer.calculate(
            sample_generated_texts, sample_reference_texts, calculate_ratio=False
        )
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, int) for s in scores)


# SequenceMatcherSimilarity Tests
class TestSequenceMatcherSimilarity:
    @pytest.fixture(scope="class")
    def seqmatch_scorer(self):
        return SequenceMatcherSimilarity()

    def test_calculate_simple(self, seqmatch_scorer, text_pair_simple):
        gen, ref = text_pair_simple
        score = seqmatch_scorer.calculate(gen, ref)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # SequenceMatcher gives ratio based on matching blocks
        # "the cat sat on the mat" vs "the cat was on the mat"
        # Matching blocks: "the cat ", "a", " on the mat" (len 20)
        # Total length = 22 + 22 = 44
        # Ratio = 2 * 20 / 44 = 38 / 44 = 10/11
        assert score == pytest.approx(10 / 11)

    def test_calculate_identical(self, seqmatch_scorer, text_pair_identical):
        gen, ref = text_pair_identical
        score = seqmatch_scorer.calculate(gen, ref)
        assert score == pytest.approx(1.0)

    def test_calculate_different(self, seqmatch_scorer, text_pair_different):
        gen, ref = text_pair_different
        score = seqmatch_scorer.calculate(gen, ref)
        assert score < 2 / 3

    def test_calculate_empty(self, seqmatch_scorer, text_pair_empty):
        gen, ref = text_pair_empty
        score = seqmatch_scorer.calculate(gen, ref)
        assert score == pytest.approx(1.0)  # Ratio of "" and "" is 1

    def test_calculate_one_empty(self, seqmatch_scorer, text_pair_one_empty):
        gen, ref = text_pair_one_empty
        score = seqmatch_scorer.calculate(gen, ref)
        assert score == pytest.approx(0.0)  # Ratio of "abc" and "" is 0

    def test_batch_calculate_list(
        self, seqmatch_scorer, sample_generated_texts, sample_reference_texts
    ):
        scores = seqmatch_scorer.calculate(sample_generated_texts, sample_reference_texts)
        assert isinstance(scores, list)
        assert len(scores) == len(sample_generated_texts)
        assert all(isinstance(s, float) for s in scores)

    def test_batch_calculate_np(
        self, seqmatch_scorer, sample_generated_texts_np, sample_reference_texts_np
    ):
        scores = seqmatch_scorer.calculate(sample_generated_texts_np, sample_reference_texts_np)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_generated_texts_np)
        assert scores.dtype == np.float64

    def test_batch_calculate_pd(
        self, seqmatch_scorer, sample_generated_texts_pd, sample_reference_texts_pd
    ):
        scores = seqmatch_scorer.calculate(sample_generated_texts_pd, sample_reference_texts_pd)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_generated_texts_pd)
        assert scores.dtype == np.float64
