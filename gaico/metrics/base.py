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
        self, generated_text: str, reference_text: str, **kwargs: Any
    ) -> float | dict:
        """
        (Internal) Calculate the metric for single pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional keyword arguments for specific metrics (additional_params, metric-specific flags etc).
        :return: The calculated metric score
        :rtype: float | dict
        """
        pass

    @abstractmethod
    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series | float | dict:
        """
        (Internal) Calculate the metric for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments for specific metrics (additional_params, metric-specific flags etc).
        :return: A list of metric scores or a single aggregated score
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series | float | dict
        """
        pass

    def calculate(
        self,
        generated_texts: str | Iterable | np.ndarray | pd.Series,
        reference_texts: Optional[str | Iterable | np.ndarray | pd.Series],
        **kwargs: Any,
    ) -> float | List[float] | dict | List[dict] | np.ndarray | pd.Series | None:
        """
        Calculates the metric for a single or batch of generated and reference texts.
        This method handles both single and batch inputs for generated and reference texts.

        If the reference text is None and generated_texts is iterable, the function will assume the first element of the iterable as the reference text. A warning will be printed.

        Additionally, the function supports the following input combinations:
            1. If both inputs are single strings, a single score is returned.
            2. If one is a string and the other an iterable, the function broadcasts the string so that it matches the length of the iterable.
            3. If both inputs are iterables, they must have the same length.
            4. If inputs are None, the function raises a ValueError.

        :param generated_texts: A single generated text or an iterable of generated texts. Must not be None or effectively empty.
        :type generated_texts: str | Iterable | np.ndarray | pd.Series
        :param reference_texts: A single reference text, an iterable of reference texts, or None. If None, empty, or effectively empty, the first generated text is used as the reference.
        :type reference_texts: Optional[str | Iterable | np.ndarray | pd.Series]
        :param kwargs: Additional keyword arguments for specific metrics (additional_params, metric-specific flags etc).
        :return: The calculated metric score(s).
        :rtype: float | List[float] | dict | List[dict] | np.ndarray | pd.Series | None
        :raises ValueError: If `generated_texts` is None or effectively empty, or if batch inputs have mismatched lengths (when reference is not derived).
        :raises TypeError: If inputs cannot be converted to suitable iterables.
        """
        if generated_texts is None:
            raise ValueError("generated_texts must be provided and cannot be None.")

        try:
            generated_iterable = to_iterable(generated_texts)
        except (TypeError, ValueError) as e:
            raise TypeError(f"generated_texts could not be converted to suitable iterables: {e}")

        if len(generated_iterable) == 0:
            raise ValueError("generated_texts cannot be an empty iterable.")
        # Check if all elements in generated_iterable are empty/whitespace (if it's a list of strings)
        # This check ensures that generated_iterable[0] is meaningful if used as a reference.
        if all(
            isinstance(g, str) and not str(g).strip() for g in generated_iterable if g is not None
        ):
            raise ValueError(
                "generated_texts cannot consist solely of empty or whitespace strings if reference is to be derived."
            )

        actual_reference_texts = reference_texts
        ref_is_missing_or_empty = False

        if actual_reference_texts is None:
            ref_is_missing_or_empty = True
        elif isinstance(actual_reference_texts, str) and not actual_reference_texts.strip():
            ref_is_missing_or_empty = True
        elif not isinstance(actual_reference_texts, str):  # It's an iterable type
            try:
                # Convert to list to make it easy to check emptiness and content
                temp_ref_list_check = list(to_iterable(actual_reference_texts))
                if not temp_ref_list_check or all(
                    isinstance(r, str) and not r.strip()
                    for r in temp_ref_list_check
                    if r is not None
                ):
                    ref_is_missing_or_empty = True
            except (TypeError, ValueError):  # If to_iterable fails
                ref_is_missing_or_empty = True

        if ref_is_missing_or_empty:
            print(
                "Warning: Reference text is missing or effectively empty. "
                "Using the first element of generated_texts as reference."
            )
            actual_reference_texts = str(generated_iterable[0])

        is_gen_str = isinstance(generated_texts, str)
        is_ref_str = isinstance(actual_reference_texts, str)

        try:
            reference_iterable = to_iterable(actual_reference_texts)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"actual_reference_texts ('{actual_reference_texts}') could not be converted to suitable iterables: {e}"
            )

        len_gen = len(generated_iterable)
        len_ref = len(reference_iterable)

        if is_gen_str and is_ref_str:
            # If both are strings, call single_calculate
            return self._single_calculate(generated_iterable[0], reference_iterable[0], **kwargs)

        elif is_gen_str and not is_ref_str:
            if len_ref == 0:
                raise ValueError(
                    "Reference iterable cannot be empty if generated text is a single string."
                )
            # If generated is a single string and reference is a list, expand
            # the single_generated text to match the number of references
            # and call batch_calculate.
            expanded_generated = [generated_iterable[0]] * len_ref
            return self._batch_calculate(expanded_generated, reference_iterable, **kwargs)

        elif not is_gen_str and is_ref_str:
            if len_gen == 0:
                raise ValueError("Generated iterable cannot be empty.")
            # If reference is a single string and generated is a list, expand
            # the single_reference text to match the number of generated texts
            # and call batch_calculate.
            expanded_reference = [reference_iterable[0]] * len_gen
            return self._batch_calculate(generated_iterable, expanded_reference, **kwargs)

        elif not is_gen_str and not is_ref_str:
            if len_gen != len_ref:
                raise ValueError(
                    f"Batch inputs: generated_texts (len {len_gen}) and reference_texts (len {len_ref}) must have the same length."
                )
            return self._batch_calculate(generated_iterable, reference_iterable, **kwargs)

        else:
            raise RuntimeError(
                "Internal error: Unhandled case in BaseMetric.calculate input dispatching."
            )
