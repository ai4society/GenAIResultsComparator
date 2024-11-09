from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from bert_score import BERTScorer

from .base import BaseMetric
from .utils import to_iterable


class BERTScore(BaseMetric):
    """
    BERTScore implementation for semantic similarity.

    This class provides methods to calculate BERTScore for individual
    sentence pairs and for batches of sentences.

    :param model_type: The BERT model to use, defaults to "bert-base-uncased"
    :type model_type: str
    :param output_val: The output value to return.
        Should be one of "precision", "recall", or "f1" to return a single score.
        Defaults to a dictionary of all scores: {"precision": P, "recall": R, "f1": F1}.
    :type output_val: Optional[str], optional
    :param num_layers: Number of layers to use from BERT, defaults to 8
    :type num_layers: int
    :param batch_size: Batch size for processing, defaults to 64
    :type batch_size: int
    :param additional_params: Additional parameters to pass to BERTScorer,
        defaults to None
    :type additional_params: Dict[str, Any], optional

    Attributes:
        scorer (BERTScorer): The BERTScorer object used for calculations
    """

    def __init__(
        self,
        model_type="bert-base-uncased",
        output_val: Optional[str] = None,
        num_layers=8,
        batch_size=64,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        params = {"model_type": model_type, "num_layers": num_layers, "batch_size": batch_size}
        if additional_params:
            params.update(additional_params)

        self.scorer = BERTScorer(**params)

        if output_val not in ["precision", "recall", "f1", None]:
            raise ValueError("output_val must be one of 'precision', 'recall', 'f1', or None")
        self.output_val = output_val

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Calculate the BERTScore for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param additional_params: Additional parameters to pass to
            the BERTScore calculation, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: A dictionary containing precision, recall, and F1 scores
        :rtype: dict
        """

        params = {}
        if additional_params:
            params.update(additional_params)

        P, R, F1 = self.scorer.score([generated_text], [reference_text], **params)

        match self.output_val:
            case "precision":
                return P.item()
            case "recall":
                return R.item()
            case "f1":
                return F1.item()
            case _:
                return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[List[dict], np.ndarray, pd.Series]:
        """
        Calculate BERTScores for a batch of generated and reference texts.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param additional_params: Additional parameters to pass to
            the BERTScore calculation, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: A list, numpy array, or pandas Series of dictionaries
            containing precision, recall, and F1 scores
        :rtype: Union[List[dict], np.ndarray, pd.Series]
        """
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = {}
        if additional_params:
            params.update(additional_params)

        P, R, F1 = self.scorer.score(list(gen_texts), list(ref_texts), **params)

        scores = [
            {"precision": p.item(), "recall": r.item(), "f1": f.item()} for p, r, f in zip(P, R, F1)
        ]

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return np.array(scores)

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return pd.Series(scores, index=gen_texts.index)

        else:
            return scores
