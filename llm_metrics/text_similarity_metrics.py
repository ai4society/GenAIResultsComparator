from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from Levenshtein import distance, ratio
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseMetric
from .utils import to_iterable


class JaccardSimilarity(BaseMetric):
    """
    Jaccard Similarity implementation for text similarity using the formula:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Supports calculation for individual sentence pairs and for batches of sentences.
    """

    def __init__(self):
        """Initialize the Jaccard Similarity metric."""
        pass

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
    ) -> float:
        """
        Calculate the Jaccard Similarity for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :return: The Jaccard Similarity score
        :rtype: float
        """
        gen_words = set(generated_text.lower().split())
        ref_words = set(reference_text.lower().split())

        intersection = len(gen_words.intersection(ref_words))
        union = len(gen_words.union(ref_words))

        return intersection / union if union > 0 else 0.0

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        """
        Calculate Jaccard Similarity for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :return: A list, numpy array, or pandas Series of Jaccard Similarity scores
        :rtype: Union[np.ndarray, pd.Series, List[float]]
        """
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return np.array(
                [self.calculate(gen, ref) for gen, ref in zip(gen_texts, ref_texts)]
            )

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return gen_texts.combine(ref_texts, lambda g, r: self.calculate(g, r))

        else:
            return [self.calculate(gen, ref) for gen, ref in zip(gen_texts, ref_texts)]


class CosineSimilarity(BaseMetric):
    """
    Cosine Similarity implementation for text similarity using `cosine_similarity` from scikit-learn.
    The class also uses the `CountVectorizer` from scikit-learn to convert text to vectors.

    Supports calculation for individual sentence pairs and for batches of sentences.
    """

    def __init__(self, additional_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Cosine Similarity metric.

        :param additional_params: Additional parameters for the CountVectorizer, defaults to None
        :type additional_params: Dict[str, Any], optional
        """
        self.vectorizer = CountVectorizer(**(additional_params or {}))

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the Cosine Similarity for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param additional_params: Additional parameters to pass to the `cosine_similarity` function, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The Cosine Similarity score
        :rtype: float
        """
        # Handle empty strings:
        if not generated_text.strip() and not reference_text.strip():
            return 1.0
        if not generated_text.strip() or not reference_text.strip():
            return 0.0

        params = {}
        if additional_params:
            params.update(additional_params)

        vectors = self.vectorizer.fit_transform([generated_text, reference_text])

        # For entirely similar text, the cosine similarity might be slightly greater than 1 due to floating point precision
        # Hence, we clip the value to be in the range [0, 1]
        similarity = cosine_similarity(vectors[0], vectors[1], **params)[0][0]
        return min(max(similarity, 0.0), 1.0)

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        """
        Calculate Cosine Similarity for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param additional_params: Additional parameters for the `cosine_similarity` function, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: A list, numpy array, or pandas Series of Cosine Similarity scores
        :rtype: Union[np.ndarray, pd.Series, List[float]]
        """

        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        # Convert to lists for easier manipulation
        gen_list = list(gen_texts)
        ref_list = list(ref_texts)

        # Handle empty strings for each pair
        results = []
        for i in range(len(gen_list)):
            gen_stripped = gen_list[i].strip() if isinstance(gen_list[i], str) else ""
            ref_stripped = ref_list[i].strip() if isinstance(ref_list[i], str) else ""

            if not gen_stripped and not ref_stripped:
                results.append(1.0)  # Both empty - perfect match
            elif not gen_stripped or not ref_stripped:
                results.append(0.0)  # One empty - no match
            else:
                # Both non-empty - calculate similarity
                params = {}
                if additional_params:
                    params.update(additional_params)

                vectors = self.vectorizer.fit_transform([gen_list[i], ref_list[i]])
                similarity = cosine_similarity(vectors[0], vectors[1], **params)[0][0]
                results.append(min(max(similarity, 0.0), 1.0))  # Clip to [0, 1]

        # Return results in the appropriate format
        if isinstance(generated_texts, np.ndarray) and isinstance(
            reference_texts, np.ndarray
        ):
            return np.array(results)
        elif isinstance(generated_texts, pd.Series) and isinstance(
            reference_texts, pd.Series
        ):
            return pd.Series(results, index=generated_texts.index)
        else:
            return results


class LevenshteinDistance(BaseMetric):
    """
    This class provides methods to calculate Levenshtein Distance for individual sentence pairs and for batches of sentences.
    It uses the `distance` and `ratio` functions from the `Levenshtein` package.
    """

    def __init__(self):
        """Initialize the Levenshtein Distance metric."""
        pass

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        calculate_ratio: bool = True,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the Levenshtein Distance for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param calculate_ratio: Whether to calculate the ratio of the distance to the length of the longer string, defaults to True.
            If True, returns the ratio, else returns the distance.
        :type calculate_ratio: bool, optional
        :param additional_params: Additional parameters for calculation, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The Levenshtein Distance or Ratio score
        :rtype: float
        """
        params = {}
        if additional_params:
            params.update(additional_params)

        if calculate_ratio:
            return ratio(generated_text, reference_text, **params)
        return distance(generated_text, reference_text, **params)

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        calculate_ratio: bool = True,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        """
        Calculate Levenshtein Distance for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param calculate_ratio: Whether to calculate the ratio of the distance to the length of the longer string, defaults to True.
            If True, returns the ratio, else returns the distance.
        :type calculate_ratio: bool, optional
        :param additional_params: Additional parameters for calculation, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: A list, numpy array, or pandas Series of Levenshtein Distance or Ratio scores
        :rtype: Union[np.ndarray, pd.Series, List[float]]
        """
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = {}
        if additional_params:
            params.update(additional_params)

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return np.array(
                [
                    self.calculate(gen, ref, calculate_ratio, params)
                    for gen, ref in zip(gen_texts, ref_texts)
                ]
            )

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return gen_texts.combine(
                ref_texts, lambda g, r: self.calculate(g, r, calculate_ratio, params)
            )

        else:
            return [
                self.calculate(gen, ref, calculate_ratio, params)
                for gen, ref in zip(gen_texts, ref_texts)
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

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the SequenceMatcher Similarity ratio for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param additional_params: Additional parameters for SequenceMatcher, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The SequenceMatcher Similarity ratio
        :rtype: float
        """
        params = {}
        if additional_params:
            params.update(additional_params)

        s_matcher = SequenceMatcher(None, generated_text, reference_text, **params)

        return s_matcher.ratio()

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        """
        Calculate SequenceMatcher Similarity for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param additional_params: Additional parameters to pass to SequenceMatcher, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: A list, numpy array, or pandas Series of SequenceMatcher Similarity ratios
        :rtype: Union[np.ndarray, pd.Series, List[float]]
        """

        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = {}
        if additional_params:
            params.update(additional_params)

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return np.array(
                [
                    self.calculate(gen, ref, params)
                    for gen, ref in zip(gen_texts, ref_texts)
                ]
            )

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return gen_texts.combine(
                ref_texts, lambda g, r: self.calculate(g, r, params)
            )

        else:
            return [
                self.calculate(gen, ref, params)
                for gen, ref in zip(gen_texts, ref_texts)
            ]
