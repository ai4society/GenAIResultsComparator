from abc import ABC
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from ..base import BaseMetric


class AudioMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate on audio data.
    Input can be various audio representations (e.g., np.array waveform, audio path).
    """

    pass


# TODO: Placeholder
class AudioSNRNormalized(AudioMetric):
    """
    Placeholder for a normalized Signal-to-Noise Ratio (SNR) metric for audio.
    SNR is often in dB. Normalization would map it to 0-1.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the AudioSNRNormalized metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate normalized SNR for a single pair of audio signals.

        :param generated_item: The generated audio (e.g., np.array waveform, path).
        :type generated_item: Any
        :param reference_item: The reference audio (often considered the 'signal' part).
        :type reference_item: Any
        :param kwargs: Additional keyword arguments.
        :return: Placeholder normalized SNR score.
        :rtype: float | dict
        """
        print("Warning: AudioSNRNormalized._single_calculate is a placeholder.")
        return 0.0  # Placeholder

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate normalized SNR for a batch of audio signals.

        :param generated_items: Iterable of generated audio.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference audio.
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of normalized SNR scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        print("Warning: AudioSNRNormalized._batch_calculate is a placeholder.")
        return []  # Placeholder


# TODO: Placeholder
class SpectrogramDistance(AudioMetric):
    """
    Placeholder for a metric calculating distance between audio spectrograms.
    (e.g., Euclidean distance, cosine distance on spectrogram features).
    """

    def __init__(self, **kwargs: Any):
        """Initialize the SpectrogramDistance metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate distance between spectrograms of two audio signals.

        :param generated_item: The generated audio.
        :type generated_item: Any
        :param reference_item: The reference audio.
        :type reference_item: Any
        :param kwargs: Additional keyword arguments for spectrogram generation or distance calculation.
        :return: Placeholder spectrogram distance score (lower is better, or normalized 0-1 for similarity).
        :rtype: float | dict
        """
        print("Warning: SpectrogramDistance._single_calculate is a placeholder.")
        return 0.0  # Placeholder

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate spectrogram distances for a batch of audio signals.

        :param generated_items: Iterable of generated audio.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference audio.
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of spectrogram distance scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        print("Warning: SpectrogramDistance._batch_calculate is a placeholder.")
        return []  # Placeholder
