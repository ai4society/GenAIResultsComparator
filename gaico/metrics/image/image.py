from abc import ABC
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from ..base import BaseMetric
from PIL import Image

class ImageMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate on image data.
    Input can be various image representations (e.g., np.array, image path).
    """

    pass

class AverageHash(ImageMetric):
    """
    Normalized average hash (aHash) similarity for images.
    This method compares 8x8 grayscale downsampled images and computes
    Hamming similarity of the resulting binary hash. The score is normalized to [0, 1].
    """

    def __init__(self, **kwargs: Any):
        """Initialize the aHash-based image similarity metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        Calculate normalized aHash similarity for a single pair of images.

        :param generated_item: The generated image (e.g., np.array, path).
        :type generated_item: Any
        :param reference_item: The reference image.
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (not used here).
        :return: Normalized aHash similarity score between 0 and 1.
        :rtype: float | dict
        """
        # Convert to PIL image if input is a NumPy array.
        if isinstance(generated_item, np.ndarray):
            generated_item = Image.fromarray(generated_item)
        if isinstance(reference_item, np.ndarray):
            reference_item = Image.fromarray(reference_item)

        # Resize to 8x8 and convert to grayscale.
        gen_resized = generated_item.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        ref_resized = reference_item.convert("L").resize((8, 8), Image.Resampling.LANCZOS)

        # Compute average pixel values.
        gen_array = np.array(gen_resized, dtype=np.float32)
        ref_array = np.array(ref_resized, dtype=np.float32)
        gen_mean = gen_array.mean()
        ref_mean = ref_array.mean()

        # Compute binary hash: 1 if pixel > mean, else 0.
        gen_hash = (gen_array > gen_mean).astype(np.uint8).flatten()
        ref_hash = (ref_array > ref_mean).astype(np.uint8).flatten()

        # Hamming similarity: proportion of matching bits.
        similarity = 1.0 - np.sum(gen_hash != ref_hash) / len(gen_hash)
        return similarity

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        Calculate normalized aHash similarity for a batch of image pairs.

        :param generated_items: Iterable of generated images.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference images.
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments (not used here).
        :return: List of normalized aHash similarity scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        results = []
        for gen, ref in zip(generated_items, reference_items):
            results.append(self._single_calculate(gen, ref, **kwargs))
        return results


class HistogramMatch(ImageMetric):
    """
    Color histogram-based similarity metric for images.
    Computes normalized histogram intersection between RGB histograms of two images.
    The output is a similarity score in the range [0, 1], where 1 means the histograms are identical.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the histogram-based similarity metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        Calculate histogram intersection similarity for a single pair of images.

        :param generated_item: The generated image (e.g., np.array, path).
        :type generated_item: Any
        :param reference_item: The reference image.
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (e.g., number of histogram bins).
        :return: Normalized histogram intersection score in the range [0, 1].
        :rtype: float | dict
        """
        # Convert to PIL image if input is a NumPy array.
        if isinstance(generated_item, np.ndarray):
            generated_item = Image.fromarray(generated_item)
        if isinstance(reference_item, np.ndarray):
            reference_item = Image.fromarray(reference_item)

        # Get number of bins for histogram, default to 256.
        bins = kwargs.get("bins", 256)

        # Convert both images to RGB and extract arrays.
        gen_arr = np.array(generated_item.convert("RGB"))
        ref_arr = np.array(reference_item.convert("RGB"))

        # Compute histogram intersection across R, G, B channels.
        intersection = 0.0
        total = 0.0
        for ch in range(3):  # Iterate over R, G, B.
            gen_hist = np.histogram(gen_arr[:, :, ch], bins=bins, range=(0, 255))[0]
            ref_hist = np.histogram(ref_arr[:, :, ch], bins=bins, range=(0, 255))[0]

            # Sum minimum of each bin across histograms.
            intersection += np.sum(np.minimum(gen_hist, ref_hist))
            total += np.sum(gen_hist)

        # Normalize similarity score to [0, 1].
        similarity = intersection / total if total > 0 else 0.0
        return similarity

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        Calculate histogram intersection similarity for a batch of image pairs.

        :param generated_items: Iterable of generated images.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference images.
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments (e.g., number of histogram bins).
        :return: List of normalized histogram intersection scores for all pairs.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        results = []
        for gen, ref in zip(generated_items, reference_items):
            results.append(self._single_calculate(gen, ref, **kwargs))
        return results



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
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate normalized SSIM for a single pair of images.

        :param generated_item: The generated image (e.g., np.array, path).
        :type generated_item: Any
        :param reference_item: The reference image.
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (e.g., data_range).
        :return: Placeholder normalized SSIM score.
        :rtype: float | dict
        """
        print("Warning: SSIMNormalized._single_calculate is a placeholder.")
        return 0.0  # Placeholder, normalized SSIM should be 0-1.

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate normalized SSIM for a batch of images.

        :param generated_items: Iterable of generated images.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference images.
        :type reference_items: Iterable | np.ndarray | pd.Series
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
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Placeholder) Calculate normalized PSNR for a single pair of images.

        :param generated_item: The generated image (e.g., np.array, path).
        :type generated_item: Any
        :param reference_item: The reference image.
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (e.g., data_range).
        :return: Placeholder normalized PSNR score.
        :rtype: float | dict
        """
        print("Warning: PSNRNormalized._single_calculate is a placeholder.")
        return 0.0  # Placeholder, normalized PSNR should be 0-1.

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        (Placeholder) Calculate normalized PSNR for a batch of images.

        :param generated_items: Iterable of generated images.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference images.
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments.
        :return: Placeholder list of normalized PSNR scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        print("Warning: PSNRNormalized._batch_calculate is a placeholder.")
        return []  # Placeholder
