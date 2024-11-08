import numpy as np
import pandas as pd
import pytest

from llm_metrics import BLEU, ROUGE, JSDivergence


@pytest.mark.parametrize("metric_class", [BLEU, ROUGE, JSDivergence])
def test_ngram_metric_single(metric_class, sample_texts):
    metric = metric_class()
    for generated, reference in sample_texts:
        score = metric.calculate(generated, reference)
        assert isinstance(
            score, (float, dict)
        ), f"{metric_class.__name__} should return float or dict"
        if isinstance(score, float):  # For BLEU and JSDivergence
            assert 0 <= score <= 1, f"{metric_class.__name__} score should be between 0 and 1"
        else:  # For ROUGE
            assert all(
                0 <= v <= 1 for v in score.values()
            ), f"{metric_class.__name__} scores should be between 0 and 1"


@pytest.fixture
def numpy_texts():
    return (
        np.array(["The quick brown fox jumps over the lazy dog", "Hello world", "Python is great"]),
        np.array(
            ["A fast brown fox leaps over a sleepy canine", "Hello Earth", "Python is awesome"]
        ),
    )


@pytest.fixture
def pandas_texts():
    return (
        pd.Series(
            ["The quick brown fox jumps over the lazy dog", "Hello world", "Python is great"]
        ),
        pd.Series(
            ["A fast brown fox leaps over a sleepy canine", "Hello Earth", "Python is awesome"]
        ),
    )


@pytest.mark.parametrize("metric_class", [BLEU, ROUGE, JSDivergence])
@pytest.mark.parametrize("texts_fixture", ["sample_texts", "numpy_texts", "pandas_texts"])
def test_ngram_metric_batch(metric_class, texts_fixture, request):
    metric = metric_class()
    texts = request.getfixturevalue(texts_fixture)

    if texts_fixture == "sample_texts":
        generated, reference = zip(*texts)
    else:
        generated, reference = texts

    scores = metric.batch_calculate(generated, reference)

    if metric_class == BLEU:
        assert isinstance(
            scores, float
        ), "BLEU batch_calculate should return a single float (corpus BLEU)"
        assert 0 <= scores <= 1, "Corpus BLEU score should be between 0 and 1"
    else:
        assert len(scores) == len(generated)
        for score in scores:
            if isinstance(score, float):
                assert 0 <= score <= 1
            elif isinstance(score, dict):
                assert all(0 <= v <= 1 for v in score.values())
            else:
                raise TypeError(f"Unexpected score type: {type(score)}")


@pytest.mark.parametrize("metric_class", [BLEU, ROUGE, JSDivergence])
def test_ngram_metric_large_dataset(metric_class, large_text_dataset):
    metric = metric_class()
    generated, reference = zip(*large_text_dataset)
    scores = metric.batch_calculate(generated, reference)

    if metric_class == BLEU:
        assert isinstance(
            scores, float
        ), "BLEU batch_calculate should return a single float (corpus BLEU)"
        assert 0 <= scores <= 1, "Corpus BLEU score should be between 0 and 1"
    else:
        assert len(scores) == len(large_text_dataset)


def test_bleu_different_n():
    bleu = BLEU(n=2)
    score = bleu.calculate("The quick brown fox", "The fast brown fox")
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_bleu_batch_sentence_level():
    bleu = BLEU(n=4)
    generated = ["The quick brown fox", "Hello world"]
    reference = ["The fast brown fox", "Hello Earth"]
    scores = bleu.batch_calculate(generated, reference, use_corpus_bleu=False)
    assert isinstance(scores, list)
    assert len(scores) == 2
    assert all(isinstance(score, float) and 0 <= score <= 1 for score in scores)


def test_bleu_batch_corpus_level():
    bleu = BLEU(n=4)
    generated = ["The quick brown fox", "Hello world"]
    reference = ["The fast brown fox", "Hello Earth"]
    score = bleu.batch_calculate(generated, reference, use_corpus_bleu=True)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_rouge_different_types():
    rouge = ROUGE(rouge_types=["rouge1", "rougeL"])
    score = rouge.calculate("The quick brown fox", "The fast brown fox")
    assert isinstance(score, dict)
    assert set(score.keys()) == {"rouge1", "rougeL"}
