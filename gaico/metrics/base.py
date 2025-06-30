from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..utils import to_iterable


class BaseMetric(ABC):
    """
    Abstract base class for all language model metrics.
    This class defines the interface that all metric classes should implement.
    The public method to be accessed is `calculate`.
    """

    @abstractmethod
    def _single_calculate(
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        (Internal) Calculate the metric for a single pair of generated and reference items.

        :param generated_item: The generated item to evaluate
        :type generated_item: Any
        :param reference_item: The reference item to compare against
        :type reference_item: Any
        :param kwargs: Additional keyword arguments for specific metrics.
        :return: The calculated metric score
        :rtype: float | dict
        """
        pass

    @abstractmethod
    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series | float | dict:
        """
        (Internal) Calculate the metric for a batch of generated and reference items.

        :param generated_items: An iterable of generated items
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: An iterable of reference items
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments for specific metrics.
        :return: A list of metric scores or a single aggregated score
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series | float | dict
        """
        pass

    def calculate(
        self,
        generated: Any,
        reference: Optional[Any],
        **kwargs: Any,
    ) -> Any:
        """
        Calculates the metric for a single or batch of generated and reference items.
        This method handles both single and batch inputs.

        If the reference is None and `generated` is an iterable, the function will assume the first element of the iterable as the reference. A warning will be printed.

        :param generated: A single generated item or an iterable of generated items. Must not be None.
        :type generated: Any
        :param reference: A single reference item, an iterable of reference items, or None.
        :type reference: Optional[Any]
        :param kwargs: Additional keyword arguments for specific metrics.
        :return: The calculated metric score(s).
        :rtype: Any
        :raises ValueError: If `generated` is None, or if batch inputs have mismatched lengths.
        :raises TypeError: If inputs cannot be converted to suitable iterables.
        """
        if generated is None:
            raise ValueError("`generated` must be provided and cannot be None.")

        # Helper to check for effective emptiness of iterables
        def is_effectively_empty(item: Any) -> bool:
            if item is None:
                return True
            if isinstance(item, str) and not item.strip():
                return True
            if hasattr(item, "__len__") and len(item) == 0:
                return True
            return False

        actual_reference = reference
        if is_effectively_empty(actual_reference):
            try:
                # Try to treat `generated` as an iterable to derive reference
                gen_iterable_for_ref = to_iterable(generated)
                if len(gen_iterable_for_ref) > 0 and not is_effectively_empty(
                    gen_iterable_for_ref[0]
                ):
                    print(
                        "Warning: Reference is missing or effectively empty. "
                        "Using the first element of `generated` as reference."
                    )
                    actual_reference = gen_iterable_for_ref[0]
                else:
                    raise ValueError(
                        "Cannot derive reference from empty or effectively empty `generated`."
                    )
            except (TypeError, ValueError):
                raise ValueError("`reference` is missing and cannot be derived from `generated`.")

        # At this point, actual_reference is guaranteed to be non-empty.
        try:
            generated_iterable = to_iterable(generated)
            reference_iterable = to_iterable(actual_reference)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Inputs could not be converted to suitable iterables: {e}")

        is_gen_single_item = len(generated_iterable) == 1 and not isinstance(
            generated, (list, tuple, pd.Series)
        )
        is_ref_single_item = len(reference_iterable) == 1 and not isinstance(
            actual_reference, (list, tuple, pd.Series)
        )

        len_gen = len(generated_iterable)
        len_ref = len(reference_iterable)

        if is_gen_single_item and is_ref_single_item:
            return self._single_calculate(generated_iterable[0], reference_iterable[0], **kwargs)

        # Broadcasting logic
        if len_gen == 1 and len_ref > 1:
            generated_iterable = [generated_iterable[0]] * len_ref
        elif len_ref == 1 and len_gen > 1:
            reference_iterable = [reference_iterable[0]] * len_gen

        if len(generated_iterable) != len(reference_iterable):
            raise ValueError(
                f"Batch inputs: generated (len {len(generated_iterable)}) and reference (len {len(reference_iterable)}) must have the same length."
            )

        return self._batch_calculate(generated_iterable, reference_iterable, **kwargs)
