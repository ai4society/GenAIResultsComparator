from abc import ABC, abstractmethod
from typing import Iterable, List, Union

import numpy as np
import pandas as pd


class BaseMetric(ABC):
    """
    Abstract base class for all language model metrics.
    This class defines the interface that all metric classes should implement.
    """

    @abstractmethod
    def calculate(
        self,
        generated_text: str,
        reference_text: str,
    ) -> Union[float, dict]:
        """
        Calculate the metric for single pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :return: The calculated metric score
        :rtype: float or dict (depending on the metric)
        """
        pass

    @abstractmethod
    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
    ) -> Union[List[float], List[dict], np.ndarray, pd.Series]:
        """
        Calculate the metric for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :return: A list of metric scores or a single aggregated score
        :rtype: Union[List[float], List[dict], float, dict]
        """
        pass
