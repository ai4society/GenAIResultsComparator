from abc import ABC, abstractmethod
from encodings.punycode import T
from typing import Iterable, List, Union , Dict , Any
import warnings

import numpy as np
import pandas as pd

from .utils import to_iterable


class BaseMetric(ABC):
    """
    Abstract base class for all language model metrics.
    This class defines the interface that all metric classes should implement.
    The public method to be accessed is `calculate`.
    """

    @abstractmethod
    def single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any 
    ) -> Union[float, dict]:
        """
        (Internal) Calculate the metric for single pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional keyword arguments for specific metrics (additional_params, metric-specific flags etc).
        :return: The calculated metric score
        :rtype: float or dict (depending on the metric)
        """
        pass

    @abstractmethod
    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        **kwargs: Any
    ) -> Union[List[float], List[dict], np.ndarray, pd.Series, float, dict]:
        """
        (Internal) Calculate the metric for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param kwargs: Additional keyword arguments for specific metrics (additional_params, metric-specific flags etc).
        :return: A list of metric scores or a single aggregated score
        :rtype: Union[List[float], List[dict], float, dict]
        """
        pass
    
    def calculate(
        self,
        generated_texts: Union[str, Iterable, np.ndarray, pd.Series],
        reference_texts: Union[str, Iterable, np.ndarray, pd.Series],
        **kwargs: Any
    ) -> Union[float, List[float], dict, List[dict], np.ndarray, pd.Series, None]:
        """
        Calculates the metric for a single or batch of generated and reference texts.
        This method handles both single and batch inputs for generated and reference texts.
        - If both inputs are single strings, it will call `single_calculate`.
        - If one is a string and the other an iterable, it broadcasts the string and calls `batch_calculate`.
        - If both inputs are iterables, they must have the same length, and it will call `batch_calculate`.
        - If inputs are None or empty, it issues a warning and returns None.
        
        :param generated_texts: A single generated text or an iterable of generated texts
        :type generated_texts: Union[str, Iterable, np.ndarray, pd.Series]
        :param reference_texts: A single reference text or an iterable of reference texts
        :type reference_texts: Union[str, Iterable, np.ndarray, pd.Series]
        :param kwargs: Additional keyword arguments for specific metrics (additional_params, metric-specific flags etc).
        :return: The calculated metric score(s) or None if inputs are invalid.
        :rtype: Union[float, List[float], dict, List[dict], np.ndarray, pd.Series, None]
        """
        is_gen_str = isinstance(generated_texts, str)
        is_ref_str = isinstance(reference_texts, str)
        
        if generated_texts is None or reference_texts is None:
            raise ValueError("Both generated_texts and reference_texts must be provided.")
        
        if is_gen_str and not generated_texts:
            warnings.warn("Generated text input is an empty string. Returning None.", UserWarning)
            return None
        if is_ref_str and not reference_texts:
            warnings.warn("Reference text input is an empty string. Returning None.", UserWarning)
            return None
        
        # Converting inputs to iterable format
        try:
            generated_iterable = to_iterable(generated_texts)
            reference_iterable = to_iterable(reference_texts)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Inputs could not be converted to suitable iterables for metrics: {e}")
        
        len_gen = len(generated_iterable)
        len_ref = len(reference_iterable)
        
        if is_gen_str and is_ref_str:
            # If both are strings, call single_calculate
            return self.single_calculate(generated_iterable[0], reference_iterable[0], **kwargs) 
        
        elif is_gen_str and not is_ref_str:
            # If generated is a single string and reference is a list, expand 
            # the single_generated text to match the number of references 
            # and call batch_calculate.
            expanded_generated = [generated_iterable[0]] * len_ref
            return self.batch_calculate(expanded_generated, reference_iterable, **kwargs)
        
        elif not is_gen_str and is_ref_str:
            # If reference is a single string and generated is a list, expand 
            # the single_reference text to match the number of generated texts 
            # and call batch_calculate.
            expanded_reference = [reference_iterable[0]] * len_gen
            return self.batch_calculate(generated_iterable, expanded_reference, **kwargs)
        
        elif not is_gen_str and not is_ref_str:
            return self.batch_calculate(generated_iterable, reference_iterable, **kwargs)
        
        else:
            raise RuntimeError("Internal error: Unhandled case in BaseMetric.calculate input dispatching.")
        
        
