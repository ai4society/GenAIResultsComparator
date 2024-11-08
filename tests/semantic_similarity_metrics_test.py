import pytest

from llm_metrics import BERTScore


@pytest.mark.parametrize("metric_class", [BERTScore])
def test_semantic_similarity_metric_single(metric_class, sample_texts):
    metric = metric_class()
    for generated, reference in sample_texts:
        score = metric.calculate(generated, reference)
        if isinstance(score, dict):
            assert all(isinstance(v, float) and 0 <= v <= 1 for v in score.values())
        else:
            assert isinstance(score, float)
            assert 0 <= score <= 1


@pytest.mark.parametrize("metric_class", [BERTScore])
def test_semantic_similarity_metric_batch(metric_class, sample_texts):
    metric = metric_class()
    generated, reference = zip(*sample_texts)
    scores = metric.batch_calculate(generated, reference)
    assert len(scores) == len(sample_texts)
    for score in scores:
        if isinstance(score, dict):
            assert all(isinstance(v, float) and 0 <= v <= 1 for v in score.values())
        else:
            assert isinstance(score, float)
            assert 0 <= score <= 1


@pytest.mark.parametrize("metric_class", [BERTScore])
def test_semantic_similarity_metric_large_dataset(metric_class, large_text_dataset):
    metric = metric_class()
    generated, reference = zip(*large_text_dataset)
    scores = metric.batch_calculate(generated, reference)
    assert len(scores) == len(large_text_dataset)


def test_bert_score_components():
    bert_score = BERTScore()
    score = bert_score.calculate("The quick brown fox", "The fast brown fox")
    assert isinstance(score, dict)
    assert set(score.keys()) == {"precision", "recall", "f1"}
