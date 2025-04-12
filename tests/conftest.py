import numpy as np
import pandas as pd
import pytest


# Sample Data
@pytest.fixture
def sample_generated_texts():
    return [
        "this is a generated text",
        "another example sentence",
        "the quick brown fox jumps over the lazy dog",
        "",  # Empty string case
        "identical text",
    ]


@pytest.fixture
def sample_reference_texts():
    return [
        "this is the reference text",
        "another reference sentence",
        "the quick brown fox jumps over the lazy dog",
        "reference for empty",  # Reference for empty generated
        "identical text",
    ]


# Data Formats
@pytest.fixture
def sample_generated_texts_np(sample_generated_texts):
    return np.array(sample_generated_texts)


@pytest.fixture
def sample_reference_texts_np(sample_reference_texts):
    return np.array(sample_reference_texts)


@pytest.fixture
def sample_generated_texts_pd(sample_generated_texts):
    return pd.Series(sample_generated_texts)


@pytest.fixture
def sample_reference_texts_pd(sample_reference_texts):
    return pd.Series(sample_reference_texts)


# Single Pair Data
@pytest.fixture
def text_pair_simple():
    return "the cat sat on the mat", "the cat was on the mat"


@pytest.fixture
def text_pair_identical():
    return "same text", "same text"


@pytest.fixture
def text_pair_different():
    return "completely one thing", "totally another thing"


@pytest.fixture
def text_pair_empty():
    return "", ""


@pytest.fixture
def text_pair_one_empty():
    return "some text", ""
