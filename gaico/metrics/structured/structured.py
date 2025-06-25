from abc import ABC
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from ..base import BaseMetric


class StructuredOutputMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate on structured text or sequences.
    Input is typically a parsed representation of the sequence or structure.
    """

    pass


class PlanningSequenceMetric(StructuredOutputMetric, ABC):
    """
    Abstract base class for metrics designed to evaluate action sequences,
    often found in automated planning outputs.
    """

    pass


class TimeSeriesDataMetric(StructuredOutputMetric, ABC):
    """
    Abstract base class for metrics evaluating time-series data,
    which can be textual or structured.
    """

    pass


# TODO: Placeholder
class ActionSequenceDiff(PlanningSequenceMetric):
    """
    Placeholder for a metric that calculates differences between action sequences.
    (e.g., edit distance for actions, alignment scores).
    """

    def __init__(self, **kwargs: Any):
        """Initialize the ActionSequenceDiff metric."""
        super().__init__(**kwargs)  # Call super if BaseMetric or parents have __init__ logic

    def _single_calculate(
        self, generated_text: Any, reference_text: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate difference for a single pair of action sequences.

        :param generated_text: The generated action sequence.
        :type generated_text: Any
        :param reference_text: The reference action sequence.
        :type reference_text: Any
        :param kwargs: Additional keyword arguments.
        :return: Placeholder score or dictionary of scores.
        :rtype: float | dict
        """
        # Placeholder: Replace with actual calculation
        print("Warning: ActionSequenceDiff._single_calculate is a placeholder and not implemented.")
        return 0.0  # Placeholder value

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate differences for a batch of action sequences.

        :param generated_texts: Iterable of generated action sequences.
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Iterable of reference action sequences.
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of scores or dictionaries.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        # Placeholder: Replace with actual batch calculation or iterative calls to _single_calculate
        print("Warning: ActionSequenceDiff._batch_calculate is a placeholder and not implemented.")
        return []  # Placeholder value


# TODO: Placeholder
class TimeSeriesElementDiff(TimeSeriesDataMetric):
    """
    Placeholder for a metric that calculates element-wise differences in time series.
    (e.g., Mean Absolute Error, Mean Squared Error on aligned series).
    """

    def __init__(self, **kwargs: Any):
        """Initialize the TimeSeriesElementDiff metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_text: Any, reference_text: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate difference for a single pair of time series.

        :param generated_text: The generated time series.
        :type generated_text: Any
        :param reference_text: The reference time series.
        :type reference_text: Any
        :param kwargs: Additional keyword arguments.
        :return: Placeholder score.
        :rtype: float | dict
        """
        print("Warning: TimeSeriesElementDiff._single_calculate is a placeholder.")
        return 0.0

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate differences for a batch of time series.

        :param generated_texts: Iterable of generated time series.
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Iterable of reference time series.
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        print("Warning: TimeSeriesElementDiff._batch_calculate is a placeholder.")
        return []  # Placeholder


# TODO: Placeholder
class DKL(TimeSeriesDataMetric):
    """
    Placeholder for a KL Divergence-based metric for time series distributions or features.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the DKL metric for time series."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_text: Any, reference_text: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate DKL for a single pair of time series representations.

        :param generated_text: Representation of the generated time series (e.g., distribution, features).
        :type generated_text: Any
        :param reference_text: Representation of the reference time series.
        :type reference_text: Any
        :param kwargs: Additional keyword arguments.
        :return: Placeholder DKL score.
        :rtype: float | dict
        """
        print("Warning: DKL (TimeSeries)._single_calculate is a placeholder.")
        return 0.0

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate DKL for a batch of time series representations.

        :param generated_texts: Iterable of generated time series representations.
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Iterable of reference time series representations.
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of DKL scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        print("Warning: DKL (TimeSeries)._batch_calculate is a placeholder.")
        return []
