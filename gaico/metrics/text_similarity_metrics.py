from difflib import SequenceMatcher
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from Levenshtein import distance, ratio

from .base import BaseMetric

# Conditional imports for scikit-learn (CosineSimilarity)
_sklearn_available = False
_CountVectorizer_cls = None
_cosine_similarity_func = None
try:
    from sklearn.feature_extraction.text import CountVectorizer as _ImportedCountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _ImportedCosineSimilarity

    _CountVectorizer_cls = _ImportedCountVectorizer
    _cosine_similarity_func = _ImportedCosineSimilarity
    _sklearn_available = True
except ImportError:
    pass  # Handled in CosineSimilarity class __init__

# This variable can be imported by tests to skip them if dependencies are missing.
__sklearn_available__ = _sklearn_available


class JaccardSimilarity(BaseMetric):
    """
    Jaccard Similarity implementation for text similarity using the formula:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Supports calculation for individual sentence pairs and for batches of sentences.
    """

    def __init__(self):
        """Initialize the Jaccard Similarity metric."""
        pass

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any,
    ) -> float:
        """
        Calculate the Jaccard Similarity for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional parameters for calculation, defaults to None.
            Note that this metric does not use any additional parameters. This parameter is included for consistency with other metrics.
        :type kwargs: Any
        :return: The Jaccard Similarity score
        :rtype: float
        """
        gen_words = set(generated_text.lower().split())
        ref_words = set(reference_text.lower().split())

        intersection = len(gen_words.intersection(ref_words))
        union = len(gen_words.union(ref_words))

        return intersection / union if union > 0 else 0.0

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> np.ndarray | pd.Series | list[float]:
        """
        Calculate Jaccard Similarity for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional parameters for calculation, defaults to None.
            Note that this metric does not use any additional parameters. This parameter is included for consistency with other metrics.
        :type kwargs: Any
        :return: A list, numpy array, or pandas Series of Jaccard Similarity scores
        :rtype: np.ndarray | pd.Series | list[float]
        """

        if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
            return np.array(
                [
                    self._single_calculate(gen, ref)
                    for gen, ref in zip(generated_texts, reference_texts)
                ]
            )

        elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
            return generated_texts.combine(
                reference_texts, lambda g, r: self._single_calculate(g, r)
            )

        else:
            return [
                self._single_calculate(gen, ref)
                for gen, ref in zip(generated_texts, reference_texts)
            ]


class CosineSimilarity(BaseMetric):
    """
    Cosine Similarity implementation for text similarity using `cosine_similarity` from scikit-learn.
    The class also uses the `CountVectorizer` from scikit-learn to convert text to vectors.

    Supports calculation for individual sentence pairs and for batches of sentences.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the Cosine Similarity metric.

        :param kwargs: Parameters for the CountVectorizer
        :type kwargs: Any
        """
        if not _sklearn_available:
            raise ImportError(
                "scikit-learn is not installed, which is required for CosineSimilarity metric. "
                "Please install it with: pip install 'gaico[cosine]'"
            )
        self.vectorizer = _CountVectorizer_cls(**kwargs)  # type: ignore

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any,
    ) -> float:
        """
        Calculate the Cosine Similarity for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional parameters to pass to the `cosine_similarity` function, defaults to None
        :type kwargs: Any
        :return: The Cosine Similarity score
        :rtype: float
        """
        # Ensure inputs are strings
        gen_str = str(generated_text)
        ref_str = str(reference_text)

        # Handle empty strings:
        if not gen_str.strip() and not ref_str.strip():
            return 1.0
        if not gen_str.strip() or not ref_str.strip():
            return 0.0

        try:
            vectors = self.vectorizer.fit_transform([gen_str, ref_str])
            vectors = cast(np.ndarray, vectors)  # Ensure vectors are numpy arrays

            # For entirely similar text, the cosine similarity might be slightly greater than 1 due to floating point precision
            # Hence, we clip the value to be in the range [0, 1]
            similarity = _cosine_similarity_func(vectors[0:1], vectors[1:2], **kwargs)[0][0]  # type: ignore
            return min(max(similarity, 0.0), 1.0)

        except ValueError as e:
            if "empty vocabulary" in str(e):
                # This means both texts, after vectorizer processing (e.g. stop word removal),
                # resulted in no usable tokens.
                # Since the initial strip checks passed, both had some content.
                # If they both become empty after processing, they are "similar" in this regard.
                return 1.0
            else:
                # Re-raise other ValueErrors
                raise ValueError(f"Error calculating cosine similarity: {str(e)}") from e

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> np.ndarray | pd.Series | list[float]:
        """
        Calculate Cosine Similarity for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional parameters for the `cosine_similarity` function, defaults to None
        :type kwargs: Any
        :return: A list, numpy array, or pandas Series of Cosine Similarity scores
        :rtype: np.ndarray | pd.Series | list[float]
        """

        # Convert to lists for easier manipulation and to ensure type consistency
        gen_list = list(map(str, generated_texts))
        ref_list = list(map(str, reference_texts))

        if len(gen_list) != len(ref_list):
            raise ValueError("Generated texts and reference texts must have the same length.")

        results = []
        for i in range(len(gen_list)):
            gen_text = gen_list[i]
            ref_text = ref_list[i]

            gen_stripped = gen_text.strip()
            ref_stripped = ref_text.strip()

            if not gen_stripped and not ref_stripped:
                results.append(1.0)  # Both empty - perfect match
            elif not gen_stripped or not ref_stripped:
                results.append(0.0)  # One empty - no match
            else:
                # Both non-empty - calculate similarity
                try:
                    vectors = self.vectorizer.fit_transform([gen_text, ref_text])
                    vectors = cast(np.ndarray, vectors)  # Ensure vectors are numpy arrays

                    similarity = _cosine_similarity_func(vectors[0:1], vectors[1:2], **kwargs)[0][0]  # type: ignore
                    results.append(min(max(similarity, 0.0), 1.0))  # Clip to [0, 1]

                except ValueError as e:
                    if "empty vocabulary" in str(e):
                        results.append(1.0)  # Both became effectively empty after vectorization
                    else:
                        raise

        # Return results in the appropriate format
        if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
            return np.array(results)
        elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
            # Try to preserve original index if it's a pandas Series
            index = generated_texts.index if hasattr(generated_texts, "index") else None
            return pd.Series(results, index=index)
        else:
            return results


class LevenshteinDistance(BaseMetric):
    """
    This class provides methods to calculate Levenshtein Distance for individual sentence pairs and for batches of sentences.
    It uses the `distance` and `ratio` functions from the `Levenshtein` package.
    """

    def __init__(self, calculate_ratio: bool = True):
        """
        Initialize the Levenshtein Distance metric.
        :param calculate_ratio: Whether to calculate the ratio of the distance to the length of the longer string, defaults to True.
        :type calculate_ratio: bool
        """
        self.calculate_ratio = calculate_ratio

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        calculate_ratio: bool = True,
        **kwargs: Any,
    ) -> float:
        """
        Calculate the Levenshtein Distance for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param calculate_ratio: Whether to calculate the ratio of the distance to the length of the longer string, defaults to True.
            If True, returns the ratio, else returns the distance.
        :type calculate_ratio: bool
        :param kwargs: Additional parameters for calculation.
        :type kwargs: Any
        :return: The Levenshtein Distance or Ratio score
        :rtype: float
        """

        if calculate_ratio or self.calculate_ratio:
            return ratio(generated_text, reference_text, **kwargs)
        return distance(generated_text, reference_text, **kwargs)

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        calculate_ratio: bool = True,
        **kwargs: Any,
    ) -> np.ndarray | pd.Series | list[float]:
        """
        Calculate Levenshtein Distance for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param calculate_ratio: Whether to calculate the ratio of the distance to the length of the longer string, defaults to True.
            If True, returns the ratio, else returns the distance.
        :type calculate_ratio: bool
        :param kwargs: Additional parameters for Levenshtein functions
        :type kwargs: Any
        :return: A list, numpy array, or pandas Series of Levenshtein Distance or Ratio scores
        :rtype: np.ndarray | pd.Series | list[float]
        """
        self.calculate_ratio = calculate_ratio

        if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
            return np.array(
                [
                    self._single_calculate(gen, ref, calculate_ratio=self.calculate_ratio, **kwargs)
                    for gen, ref in zip(generated_texts, reference_texts)
                ]
            )

        elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
            return generated_texts.combine(
                reference_texts,
                lambda g, r: self._single_calculate(
                    g, r, calculate_ratio=self.calculate_ratio, **kwargs
                ),
            )

        else:
            return [
                self._single_calculate(gen, ref, calculate_ratio=self.calculate_ratio, **kwargs)
                for gen, ref in zip(generated_texts, reference_texts)
            ]


class SequenceMatcherSimilarity(BaseMetric):
    """
    This class calculates similarity ratio between texts using the ratio() method from difflib.SequenceMatcher,
    which returns a float in the range [0, 1] indicating how similar the sequences are.

    Supports calculation for individual sentence pairs and for batches of sentences.
    """

    def __init__(self):
        """Initialize the SequenceMatcher Similarity metric"""
        pass

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any,
    ) -> float:
        """
        Calculate the SequenceMatcher Similarity ratio for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional parameters for SequenceMatcher
        :type kwargs: Any
        :return: The SequenceMatcher Similarity ratio
        :rtype: float
        """

        s_matcher = SequenceMatcher(None, generated_text, reference_text, **kwargs)

        return s_matcher.ratio()

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> np.ndarray | pd.Series | list[float]:
        """
        Calculate SequenceMatcher Similarity for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional parameters for SequenceMatcher
        :type kwargs: Any
        :return: A list, numpy array, or pandas Series of SequenceMatcher Similarity ratios
        :rtype: np.ndarray | pd.Series | list[float]
        """

        if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
            return np.array(
                [
                    self._single_calculate(gen, ref, **kwargs)
                    for gen, ref in zip(generated_texts, reference_texts)
                ]
            )

        elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
            return generated_texts.combine(
                reference_texts, lambda g, r: self._single_calculate(g, r, **kwargs)
            )

        else:
            return [
                self._single_calculate(gen, ref, **kwargs)
                for gen, ref in zip(generated_texts, reference_texts)
            ]
