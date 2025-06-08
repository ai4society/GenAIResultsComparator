from typing import Any, Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd

from .base import BaseMetric

# Conditionally import bert_score and torch
_BERTScorer_cls = None
_Tensor_cls = None
_semantic_deps_available = False

try:
    from bert_score import BERTScorer as _ImportedBERTScorer  # type: ignore

    _BERTScorer_cls = _ImportedBERTScorer
    # torch is a dependency of bert_score
    from torch import Tensor as _ImportedTensor  # type: ignore

    _Tensor_cls = _ImportedTensor
    _semantic_deps_available = True
except ImportError:
    # bert_score or its dependencies (like torch) are not installed.
    # The BERTScore class __init__ will handle raising an error.
    pass

# This variable can be imported by tests to skip them if dependencies are missing.
__semantic_deps_available__ = _semantic_deps_available


class BERTScore(BaseMetric):
    """
    This class provides methods to calculate BERTScore for individual sentence pairs and for batches of sentences.
    It uses the BERTScore library to calculate precision, recall, and F1 scores.
    """

    def __init__(
        self,
        model_type="bert-base-uncased",
        output_val: Optional[List[str]] = None,
        num_layers=8,
        batch_size=64,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the BERTScore metric.

        :param model_type: The BERT model to use, defaults to "bert-base-uncased"
        :type model_type: str
        :param output_val: The output value to return, defaults to None
            Should be one of "precision", "recall", or "f1" to return a single score of type float
            Wrap in a list to return multiple scores of type dict
            Default returns a dictionary of all scores. Equivalent to passing ["precision", "recall", "f1"]
        :type output_val: Optional[List[str]]
        :param num_layers: Number of layers to use from BERT, defaults to 8
        :type num_layers: int
        :param batch_size: Batch size for processing, defaults to 64
        :type batch_size: int
        :param additional_params: Additional parameters to pass to the BERTScorer class from the bert_score library, defaults to None
            Default only passes the model_type, num_layers, and batch_size
        :type additional_params: Dict[str, Any]
        """
        params = {
            "model_type": model_type,
            "num_layers": num_layers,
            "batch_size": batch_size,
        }
        if additional_params:
            params.update(additional_params)

        # Check if output_val is valid
        if output_val:
            if not isinstance(output_val, list):  # Check if it is a list
                raise ValueError("`output_val` must be a list")

            elif not all(val in ["precision", "recall", "f1"] for val in output_val):
                raise ValueError("`output_val` must be one of ['precision', 'recall', 'f1']")

        # Ensure output_val is a list
        self.output_val = output_val or ["precision", "recall", "f1"]

        # Now check for dependencies before initializing the scorer
        if not _semantic_deps_available:
            raise ImportError(
                "BERTScore dependencies (bert-score, torch) are not installed. "
                "Please install them with: pip install 'gaico[bertscore]'"
            )

        self.scorer = _BERTScorer_cls(**params)  # type: ignore

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any,
    ) -> dict[str, float] | float:
        """
        Calculate the BERTScore for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional parameters to pass to the score method of the BERTScorer class, defaults to None
        :type kwargs: Any
        :return: Either a single score or a dictionary of scores containing precision, recall, and F1
        :rtype: dict[str, float] | float
        """

        P: _Tensor_cls  # type: ignore
        R: _Tensor_cls  # type: ignore
        F1: _Tensor_cls  # type: ignore
        P, R, F1 = cast(
            tuple[_Tensor_cls, _Tensor_cls, _Tensor_cls],  # type: ignore
            self.scorer.score([generated_text], [reference_text], **kwargs),  # type: ignore
        )
        assert P is not None and R is not None and F1 is not None, (
            "BERTScore tensors should not be None"
        )
        out_dict = {"precision": P.item(), "recall": R.item(), "f1": F1.item()}

        # Return based on output_val
        # If a single value is requested, return that value
        # Else, if multiple values are requested, return all those values only
        # If no value is requested, return all values
        if len(self.output_val) == 1:
            return out_dict.get(self.output_val[0], 0.0)
        else:
            return {key: out_dict[key] for key in self.output_val}

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> list[float] | list[dict] | np.ndarray | pd.Series:
        """
        Calculate BERTScores for a batch of generated and reference texts.
        Supports iterables, numpy arrays, and pandas Series as input and output.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional parameters to pass to the score method of the BERTScorer class
        :type kwargs: Any
        :return: A list, numpy array, or pandas Series of dictionaries containing precision, recall, and F1 scores.
            If `output_val` is set to a single value, returns a list, numpy array, or pandas Series of that value.
        :rtype: list[float] | list[dict] | np.ndarray | pd.Series
        """
        P: _Tensor_cls  # type: ignore
        R: _Tensor_cls  # type: ignore
        F1: _Tensor_cls  # type: ignore
        P, R, F1 = cast(
            tuple[_Tensor_cls, _Tensor_cls, _Tensor_cls],  # type: ignore
            self.scorer.score(list(generated_texts), list(reference_texts), **kwargs),  # type: ignore
        )

        # Convert tensors to lists
        scores: List[dict[str, float]] = []
        p: _Tensor_cls  # type: ignore
        r: _Tensor_cls  # type: ignore
        f: _Tensor_cls  # type: ignore
        for p, r, f in zip(P, R, F1):
            assert p is not None and r is not None and f is not None, (
                "BERTScore tensors should not be None"
            )
            score_dict = {"precision": p.item(), "recall": r.item(), "f1": f.item()}
            scores.append(score_dict)

        # Define final scores based on output_val
        scores = [{key: score[key] for key in self.output_val} for score in scores]

        if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
            return np.array(scores)

        elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
            return pd.Series(scores, index=generated_texts.index)

        else:
            return scores
