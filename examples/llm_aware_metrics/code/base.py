from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Union

from numpy import ndarray
from pandas import Series

from llm_metrics.base import BaseMetric


class LLMAwareMetric(BaseMetric):
    """Base class for LLM-aware metrics that consider prompts and metadata."""

    @abstractmethod
    def calculate_with_prompt(
        self,
        text1: str,
        text2: str,
        prompt1: str,
        prompt2: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate similarity considering prompts and metadata.

        :param text1: First text to compare.
        :type text1: str
        :param text2: Second text to compare.
        :type text2: str
        :param prompt1: Prompt for the first text.
        :type prompt1: str
        :param prompt2: Prompt for the second text.
        :type prompt2: str
        :param metadata: Optional metadata for the comparison.
        :type metadata: Dict[str, Any]
        :return: Similarity score between the two texts.
        :rtype: float
        """
        pass

    @abstractmethod
    def batch_calculate_with_prompt(
        self,
        texts1: List[str],
        texts2: List[str],
        prompts1: List[str],
        prompts2: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[List[float], List[Dict[str, float]]]:
        """
        Calculate similarity for batches of texts with prompts.

        :param texts1: List of first texts to compare.
        :type texts1: List[str]
        :param texts2: List of second texts to compare.
        :type texts2: List[str]
        :param prompts1: List of prompts for the first texts.
        :type prompts1: List[str]
        :param prompts2: List of prompts for the second texts.
        :type prompts2: List[str]
        :param metadata: Optional metadata for the comparison.
        :type metadata: Dict[str, Any]
        :return: List of similarity scores between the text pairs.
        :rtype: List[float]
        """
        pass

    def calculate(self, text1: str, text2: str) -> float:
        """
        Calculate similarity without prompts.
        Calls the base metric's calculate method.

        :param text1: First text to compare.
        :type text1: str
        :param text2: Second text to compare.
        :type text2: str
        :return: Similarity score between the two texts.
        :rtype: float
        """
        raise Warning(
            "Using base comparison without prompt. "
            "Consider using calculate_with_prompt instead."
            "Else consider using the base metric directly."
        )

    def batch_calculate(
        self, texts1: Union[Iterable, ndarray, Series], texts2: Union[Iterable, ndarray, Series]
    ) -> Union[List[float], List[dict], ndarray, Series]:
        """
        Calculate similarity for batches of texts without prompts.
        Calls the base metric's batch_calculate method.

        :param texts1: List of first texts to compare.
        :type texts1: List[str]
        :param texts2: List of second texts to compare.
        :type texts2: List[str]
        :return: List of similarity scores between the text pairs.
        :rtype: List[float]
        """
        raise Warning(
            "Using base comparison without prompt. "
            "Consider using batch_calculate_with_prompt instead."
            "Else consider using the base metric directly."
        )
