import pytest


@pytest.fixture
def sample_texts():
    return [
        (
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy canine",
        ),
        ("Hello world", "Hello Earth"),
        ("Python is great", "Python is awesome"),
        ("Repeated words words words", "Different repeated words words"),
        ("Same words yet again", "Same words yet again"),
    ]


@pytest.fixture
def large_text_dataset():
    import numpy as np

    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    return [
        (" ".join(np.random.choice(words, size=100)), " ".join(np.random.choice(words, size=100)))
        for _ in range(1000)
    ]
