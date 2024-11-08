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
    Jaccard Similarity implementation for text similarity.

    This class provides methods to calculate Jaccard Similarity for individual
    sentence pairs and for batches of sentences.

    :param additional_params: Additional parameters for calculation,
        defaults to None
    :type additional_params: Dict[str, Any], optional

    Attributes:
        additional_params (Dict[str, Any]): Additional parameters
    """

    def __init__(self, additional_params: Optional[Dict[str, Any]] = None):
        self.additional_params = additional_params or {}

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the Jaccard Similarity for a
            pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param additional_params: Additional parameters for calculation
            defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The Jaccard Similarity score
        :rtype: float
        """
        params = self.additional_params.copy()
        if additional_params:
            params.update(additional_params)

        gen_words = set(generated_text.lower().split())
        ref_words = set(reference_text.lower().split())

        intersection = len(gen_words.intersection(ref_words))
        union = len(gen_words.union(ref_words))

        return intersection / union if union > 0 else 0.0

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        """
        Calculate Jaccard Similarity for a
            batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param additional_params: Additional parameters for calculation,
            defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: A list, numpy array, or pandas Series of
            Jaccard Similarity scores
        :rtype: Union[np.ndarray, pd.Series, List[float]]
        """
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = self.additional_params.copy()
        if additional_params:
            params.update(additional_params)

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return np.array(
                [self.calculate(gen, ref, params) for gen, ref in zip(gen_texts, ref_texts)]
            )

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return gen_texts.combine(ref_texts, lambda g, r: self.calculate(g, r, params))

        else:
            return [self.calculate(gen, ref, params) for gen, ref in zip(gen_texts, ref_texts)]


class CosineSimilarity(BaseMetric):
    def __init__(self, additional_params: Optional[Dict[str, Any]] = None):
        self.vectorizer = CountVectorizer(**(additional_params or {}))

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        params = {}
        if additional_params:
            params.update(additional_params)

        vectors = self.vectorizer.fit_transform([generated_text, reference_text])
        return cosine_similarity(vectors[0], vectors[1], **params)[0][0]

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = {}
        if additional_params:
            params.update(additional_params)

        vectors = self.vectorizer.fit_transform(list(gen_texts) + list(ref_texts))
        gen_vectors = vectors[: len(gen_texts)]
        ref_vectors = vectors[len(gen_texts) :]

        similarities = cosine_similarity(gen_vectors, ref_vectors, **params)

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return similarities.diagonal()

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return pd.Series(similarities.diagonal(), index=gen_texts.index)

        else:
            return similarities.diagonal().tolist()


class LevenshteinDistance(BaseMetric):
    """
    Levenshtein Distance implementation for text similarity.

    This class provides methods to calculate Levenshtein Distance for
    individual sentence pairs and for batches of sentences.

    :param additional_params: Additional parameters for calculation,
        defaults to None
    :type additional_params: Dict[str, Any], optional

    Attributes:
        additional_params (Dict[str, Any]): Additional parameters
    """

    def __init__(self, additional_params: Optional[Dict[str, Any]] = None):
        self.additional_params = additional_params or {}

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        calculate_ratio: bool = True,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the Levenshtein Distance for a pair of
            generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param calculate_ratio: Whether to calculate the ratio of the distance
            to the length of the longer string, defaults to True.
            If True, returns the ratio, else returns the distance.
        :type calculate_ratio: bool, optional
        :param additional_params: Additional parameters for calculation,
            defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The Levenshtein Distance or Ratio score
        :rtype: float
        """
        params = self.additional_params.copy()
        if additional_params:
            params.update(additional_params)

        if calculate_ratio:
            return ratio(generated_text, reference_text, **params)
        else:
            return distance(generated_text, reference_text, **params)

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        calculate_ratio: bool = True,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        """
        Calculate Levenshtein Distance for a batch of
            generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param calculate_ratio: Whether to calculate the ratio of the distance
            to the length of the longer string, defaults to True.
            If True, returns the ratio, else returns the distance.
        :type calculate_ratio: bool, optional
        :param additional_params: Additional parameters for calculation,
            defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: A list, numpy array, or pandas Series of
            Levenshtein Distance or Ratio scores
        :rtype: Union[np.ndarray, pd.Series, List[float]]
        """
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = self.additional_params.copy()
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
