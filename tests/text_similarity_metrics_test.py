import pytest

from llm_metrics import CosineSimilarity, JaccardSimilarity, LevenshteinDistance


@pytest.mark.parametrize("metric_class", [JaccardSimilarity, CosineSimilarity, LevenshteinDistance])
def test_text_similarity_metric_single(metric_class, sample_texts):
    metric = metric_class()
    for generated, reference in sample_texts:
        score = metric.calculate(generated, reference)
        assert isinstance(
            score, (float, int)
        ), f"{metric_class.__name__} should return float or int"
        if metric_class != LevenshteinDistance:
            assert 0 <= score <= 1, f"{metric_class.__name__} score should be between 0 and 1"


@pytest.mark.parametrize("metric_class", [JaccardSimilarity, CosineSimilarity, LevenshteinDistance])
def test_text_similarity_metric_batch(metric_class, sample_texts):
    metric = metric_class()
    generated, reference = zip(*sample_texts)
    scores = metric.batch_calculate(generated, reference)
    assert len(scores) == len(sample_texts)
    for score in scores:
        assert isinstance(score, (float, int))
        if metric_class != LevenshteinDistance:
            assert 0 <= score <= 1


@pytest.mark.parametrize("metric_class", [JaccardSimilarity, CosineSimilarity, LevenshteinDistance])
def test_text_similarity_metric_large_dataset(metric_class, large_text_dataset):
    metric = metric_class()
    generated, reference = zip(*large_text_dataset)
    scores = metric.batch_calculate(generated, reference)
    assert len(scores) == len(large_text_dataset)


def test_levenshtein_distance_exact_match():
    ld = LevenshteinDistance()
    assert ld.calculate("hello", "hello") == 0
