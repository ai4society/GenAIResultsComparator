from abc import ABC

from ..metrics.base import BaseMetric


class TextualMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate primarily on textual inputs (strings).

    This class serves as a common parent for various text-based comparison metrics
    like Jaccard Similarity, ROUGE, BERTScore, etc. Concrete implementations
    are expected to process string inputs for generated and reference texts.
    """

    pass
