from abc import ABC
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from ..base import BaseMetric


class ImageMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate on image data.
    Input can be various image representations (e.g., np.array, image path).
    """

    pass


# TODO: Placeholder
class SSIMNormalized(ImageMetric):
    """
    Placeholder for a normalized Structural Similarity Index (SSIM) metric for images.
    SSIM typically ranges from -1 to 1. Normalized version aims for 0 to 1.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the SSIMNormalized metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_text: Any, reference_text: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate normalized SSIM for a single pair of images.

        :param generated_text: The generated image (e.g., np.array, path).
        :type generated_text: Any
        :param reference_text: The reference image.
        :type reference_text: Any
        :param kwargs: Additional keyword arguments (e.g., data_range).
        :return: Placeholder normalized SSIM score.
        :rtype: float | dict
        """
        print("Warning: SSIMNormalized._single_calculate is a placeholder.")
        return 0.0  # Placeholder, normalized SSIM should be 0-1.

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate normalized SSIM for a batch of images.

        :param generated_texts: Iterable of generated images.
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Iterable of reference images.
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of normalized SSIM scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        print("Warning: SSIMNormalized._batch_calculate is a placeholder.")
        return []  # Placeholder


# TODO: Placeholder
class PSNRNormalized(ImageMetric):
    """
    Placeholder for a normalized Peak Signal-to-Noise Ratio (PSNR) metric for images.
    PSNR is often in dB. Normalization would map it to 0-1.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the PSNRNormalized metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_text: Any, reference_text: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate normalized PSNR for a single pair of images.

        :param generated_text: The generated image (e.g., np.array, path).
        :type generated_text: Any
        :param reference_text: The reference image.
        :type reference_text: Any
        :param kwargs: Additional keyword arguments (e.g., data_range).
        :return: Placeholder normalized PSNR score.
        :rtype: float | dict
        """
        print("Warning: PSNRNormalized._single_calculate is a placeholder.")
        return 0.0  # Placeholder, normalized PSNR should be 0-1.

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate normalized PSNR for a batch of images.

        :param generated_texts: Iterable of generated images.
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Iterable of reference images.
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of normalized PSNR scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        print("Warning: PSNRNormalized._batch_calculate is a placeholder.")
        return []  # Placeholder
