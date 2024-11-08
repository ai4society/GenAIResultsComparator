import numpy as np
import pandas as pd

from llm_metrics.utils import get_ngrams, to_iterable


def test_to_iterable():
    assert list(to_iterable([1, 2, 3])) == [1, 2, 3]
    assert list(to_iterable((1, 2, 3))) == [1, 2, 3]
    assert list(to_iterable({1, 2, 3})) == [1, 2, 3]
    assert list(to_iterable(pd.Series([1, 2, 3]))) == [1, 2, 3]
    assert list(to_iterable(np.array([1, 2, 3]))) == [1, 2, 3]
    assert list(to_iterable({"a": 1, "b": 2, "c": 3}.values())) == [1, 2, 3]
    assert list(to_iterable("test")) == ["test"]
    assert list(to_iterable(123)) == [123]


def test_get_ngrams():
    text = "the quick brown fox"
    unigrams = get_ngrams(text, 1)
    bigrams = get_ngrams(text, 2)
    trigrams = get_ngrams(text, 3)

    assert unigrams == {"the": 1, "quick": 1, "brown": 1, "fox": 1}
    assert bigrams == {"the quick": 1, "quick brown": 1, "brown fox": 1}
    assert trigrams == {"the quick brown": 1, "quick brown fox": 1}


def test_get_ngrams_empty_string():
    assert get_ngrams("", 1) == {}
    assert get_ngrams("", 2) == {}


def test_get_ngrams_single_word():
    assert get_ngrams("word", 1) == {"word": 1}
    assert get_ngrams("word", 2) == {}
