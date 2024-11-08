from typing import Any, Dict, Iterable, List, Optional, Union

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
from scipy.spatial.distance import jensenshannon

from .base import BaseMetric
from .utils import to_iterable


class BLEU(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) score implementation.

    This class provides methods to calculate BLEU scores for individual
    sentence pairs and for batches of sentences. It supports both
    sentence-level and corpus-level BLEU calculations.

    :param n: The max n-gram order to use for BLEU calculation, defaults to 4
    :type n: int
    :param smoothing_function: The smoothing function to use
        for BLEU calculation, defaults to None
    :type smoothing_function: callable, optional
    :param additional_params: Additional parameters to pass to the
        BLEU calculation, defaults to None
    :type additional_params: Dict[str, Any], optional

    Attributes:
        n (int): The maximum n-gram order used in BLEU calculation
        weights (tuple): The weights for each n-gram order
        smoothing_function (callable): The smoothing function to use for BLEU
        additional_params (Dict[str, Any]): Additional parameters to
            pass to BLEU calculation
    """

    def __init__(
        self,
        n: int = 4,
        smoothing_function=None,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        self.n = n
        self.weights = tuple([1 / n] * n)
        self.smoothing_function = smoothing_function or SmoothingFunction().method1
        self.additional_params = additional_params or {}

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the BLEU score for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param additional_params: Additional parameters to pass to the
            BLEU calculation, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The BLEU score for the text pair
        :rtype: float
        """
        params = self.additional_params.copy()
        if additional_params:
            params.update(additional_params)

        # Split texts into words and calculate sentence-level BLEU score
        return float(
            sentence_bleu(
                [reference_text.split()],
                generated_text.split(),
                weights=self.weights,
                smoothing_function=self.smoothing_function,
                **params,
            )
        )

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        use_corpus_bleu: bool = True,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[float, List[float], np.ndarray, pd.Series]:
        """
        Calculate BLEU scores for a batch of generated and reference texts.

        This method supports two modes of calculation:
        1. Corpus-level BLEU (default):
            Calculates a single BLEU score for the entire corpus.
        2. Sentence-level BLEU:
            Calculates individual BLEU scores for each sentence pair.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param use_corpus_bleu: Whether to use corpus-level BLEU calculation,
            defaults to True
        :type use_corpus_bleu: bool
        :param additional_params: Additional parameters to pass to BLEU,
            defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: Either a single corpus-level BLEU score or
            a list/array/series of sentence-level BLEU scores
        :rtype: Union[float, List[float], np.ndarray, pd.Series]
        """
        # Convert inputs to appropriate iterable types,
        # preserving numpy arrays and pandas Series
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = self.additional_params.copy()
        if additional_params:
            params.update(additional_params)

        if use_corpus_bleu:
            # Prepare data for corpus_bleu calculation
            if isinstance(gen_texts, (np.ndarray, pd.Series)):
                # For numpy arrays and pandas Series, use vectorized operations
                hypotheses = [text.split() for text in gen_texts]
                references = [[text.split()] for text in ref_texts]
            else:
                # For other iterable types, we use a list comprehension
                hypotheses = [gen.split() for gen in gen_texts]
                references = [[ref.split()] for ref in ref_texts]

            # Calculate and return the corpus-level BLEU score
            return corpus_bleu(
                references,
                hypotheses,
                weights=self.weights,
                smoothing_function=self.smoothing_function,
                **params,
            )
        else:
            # Calculate individual BLEU scores for each sentence pair
            if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
                # Use numpy's vectorization for faster calculation
                return np.array(
                    [
                        self.calculate(gen, ref, additional_params)
                        for gen, ref in zip(gen_texts, ref_texts)
                    ]
                )
            elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
                # Use pandas' apply method for Series
                return gen_texts.combine(
                    ref_texts, lambda g, r: self.calculate(g, r, additional_params)
                )
            else:
                # For other iterable types, use a list comprehension
                return [
                    self.calculate(gen, ref, additional_params)
                    for gen, ref in zip(gen_texts, ref_texts)
                ]


class ROUGE(BaseMetric):
    def __init__(
        self,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        params = {"use_stemmer": use_stemmer}
        if additional_params:
            params.update(additional_params)

        self.scorer = rouge_scorer.RougeScorer(rouge_types, **params)
        self.rouge_types = rouge_types

    def calculate(self, generated_text: str, reference_text: str) -> dict:
        scores = self.scorer.score(reference_text, generated_text)
        return {k: v.fmeasure for k, v in scores.items()}

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
    ) -> List[dict]:
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return [self.calculate(gen, ref) for gen, ref in zip(gen_texts, ref_texts)]

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return gen_texts.combine(ref_texts, self.calculate).tolist()

        else:
            return [self.calculate(gen, ref) for gen, ref in zip(gen_texts, ref_texts)]


class JSDivergence(BaseMetric):
    def __init__(self, additional_params: Optional[Dict[str, Any]] = None):
        self.additional_params = additional_params or {}

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        params = self.additional_params.copy()
        if additional_params:
            params.update(additional_params)

        gen_freq = nltk.FreqDist(generated_text.split())
        ref_freq = nltk.FreqDist(reference_text.split())
        all_words = set(gen_freq.keys()) | set(ref_freq.keys())

        gen_probs = [gen_freq.freq(word) for word in all_words]
        ref_probs = [ref_freq.freq(word) for word in all_words]

        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nan_to_num(jensenshannon(gen_probs, ref_probs, **params)))

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = self.additional_params.copy()
        if additional_params:
            params.update(additional_params)

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return np.array(
                [self.calculate(gen, ref, params) for gen, ref in zip(gen_texts, ref_texts)]
            )

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return gen_texts.combine(ref_texts, lambda g, r: self.calculate(g, r, params))

        else:
            return [self.calculate(gen, ref, params) for gen, ref in zip(gen_texts, ref_texts)]
