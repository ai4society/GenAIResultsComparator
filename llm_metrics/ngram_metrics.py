from typing import Any, Callable, Dict, Iterable, List, Optional, Union

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
    This class provides methods to calculate BLEU scores for individual sentence pairs and for batches of sentences.
    It uses the NLTK library to calculate BLEU scores.
    """

    def __init__(
        self,
        n: int = 4,
        smoothing_function: Union[Callable, SmoothingFunction] = SmoothingFunction().method1,
    ):
        """
        Initialize the BLEU scorer with the specified parameters.

        :param n: The max n-gram order to use for BLEU calculation, defaults to 4
        :type n: int
        :param smoothing_function: The smoothing function to use for BLEU, defaults to SmoothingFunction.method1 from NLTK
        :type smoothing_function: Union[Callable, SmoothingFunction], optional
        """
        self.n = n
        self.weights = tuple([1 / n] * n)
        self.smoothing_function = smoothing_function

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
        :param additional_params: Additional parameters to pass to the BLEU calculation, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The BLEU score for the text pair
        :rtype: float
        """
        params = {}
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
            Calculates a single BLEU score for the entire corpus. Uses the `corpus_bleu` method from NLTK
        2. Sentence-level BLEU:
            Calculates individual BLEU scores for each sentence pair. Uses the `sentence_bleu` method from NLTK

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param use_corpus_bleu: Whether to use corpus-level BLEU calculation, defaults to True
        :type use_corpus_bleu: bool
        :param additional_params: Additional parameters to pass to BLEU, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: Either a single corpus-level BLEU score or a list/array/series of sentence-level BLEU scores
        :rtype: Union[float, List[float], np.ndarray, pd.Series]
        """
        # Convert inputs to appropriate iterable types,
        # preserving numpy arrays and pandas Series
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = {}
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
                # Use numpy vectorization for faster calculation
                return np.array(
                    [
                        self.calculate(gen, ref, additional_params=params)
                        for gen, ref in zip(gen_texts, ref_texts)
                    ]
                )
            elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
                # Use pandas' apply method for Series
                return gen_texts.combine(
                    ref_texts, lambda g, r: self.calculate(g, r, additional_params=params)
                )
            else:
                # For other iterable types, use a list comprehension
                return [
                    self.calculate(gen, ref, additional_params=params)
                    for gen, ref in zip(gen_texts, ref_texts)
                ]


class ROUGE(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score implementation using the `rouge_score` library.
    """

    def __init__(
        self,
        rouge_types: Optional[List[str]] = None,
        use_stemmer: bool = True,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ROUGE scorer with the specified ROUGE types and parameters.

        :param rouge_types: The ROUGE types to calculate, defaults to None
            Should be one of "rouge1", "rouge2", or "rougeL" in a list to return a single F1 score of that type.
            If multiple types are provided in a list, the output will be a dictionary of F1 scores for each type.
            Defaults is None which returns a dictionary of all scores. Equivalent of passing ["rouge1", "rouge2", "rougeL"]
        :type rouge_types: Optional[List[str]]
        :param use_stemmer: Whether to use stemming for ROUGE calculation, defaults to True
        :type use_stemmer: bool
        :param additional_params: Additional parameters to pass to the ROUGE calculation, defaults to None
            Default only passes the `use_stemmer` parameter
        :type additional_params: Dict[str, Any], optional
        """
        params = {"use_stemmer": use_stemmer}
        if additional_params:
            params.update(additional_params)

        # Check if rouge_types is valid
        if rouge_types:
            if not isinstance(rouge_types, list):
                raise ValueError("rouge_types must be a list")
            elif not all(val in ["rouge1", "rouge2", "rougeL"] for val in rouge_types):
                raise ValueError("rouge_types must be one of ['rouge1', 'rouge2', 'rougeL']")

        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]

        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, **params)

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
    ) -> Union[Dict[str, float], float]:
        """
        Calculate the ROUGE score for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :return: Either a single score or a dictionary of scores containing ROUGE types
        :rtype: Union[dict, float]
        """
        scores = self.scorer.score(reference_text, generated_text)
        score_dict = {k: v.fmeasure for k, v in scores.items()}

        # Return based on the supplied ROUGE types
        if len(self.rouge_types) == 1:
            return score_dict.get(self.rouge_types[0], 0.0)
        return {key: score_dict[key] for key in self.rouge_types}

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
    ) -> Union[List[float], List[dict], np.ndarray, pd.Series]:
        """
        Calculate ROUGE scores for a batch of generated and reference texts.
        Supports iterables, numpy arrays, and pandas Series as input and output.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :return: A list, numpy array, or pandas Series of Dictionary of ROUGE scores
            If `rouge_types` is a single type, returns a list, numpy array, or pandas Series of that value.
        :rtype: Union[List[dict], np.ndarray, pd.Series]
        """
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        # `self.scorer` takes care of calculating ROUGE scores based on the supplied ROUGE types
        scores = [self.calculate(gen, ref) for gen, ref in zip(gen_texts, ref_texts)]

        if isinstance(gen_texts, np.ndarray) and isinstance(ref_texts, np.ndarray):
            return np.array(scores)

        elif isinstance(gen_texts, pd.Series) and isinstance(ref_texts, pd.Series):
            return pd.Series(scores, index=gen_texts.index)

        else:
            return scores


class JSDivergence(BaseMetric):
    """
    Jensen-Shannon Divergence metric implementation using the `scipy` library.
    """

    def __init__(
        self,
    ):
        """
        Initialize the Jensen-Shannon Divergence metric.
        """
        pass

    def calculate(
        self,
        generated_text: str,
        reference_text: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the Jensen-Shannon Divergence between a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param additional_params: Additional parameters to pass to the score method of the BERTScorer class, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return: The Jensen-Shannon Divergence score for the text pair
        :rtype: float
        """
        params = {}
        if additional_params:
            params.update(additional_params)

        gen_freq = nltk.FreqDist(generated_text.split())
        ref_freq = nltk.FreqDist(reference_text.split())
        all_words = set(gen_freq.keys()) | set(ref_freq.keys())

        gen_probs = [gen_freq.freq(word) for word in all_words]
        ref_probs = [ref_freq.freq(word) for word in all_words]

        # Handle the case where arrays are zeros
        if all(x == 0 for x in ref_probs) or all(x == 0 for x in gen_probs):
            return 0.0

        return 1.0 - float((jensenshannon(gen_probs, ref_probs, **params)))

    def batch_calculate(
        self,
        generated_texts: Union[Iterable, np.ndarray, pd.Series],
        reference_texts: Union[Iterable, np.ndarray, pd.Series],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, pd.Series, List[float]]:
        """
        Calculate Jensen-Shannon Divergence scores for a batch of generated and reference texts.
        Supports iterables, numpy arrays, and pandas Series as input and output.

        :param generated_texts: Generated texts
        :type generated_texts: Union[Iterable, np.ndarray, pd.Series]
        :param reference_texts: Reference texts
        :type reference_texts: Union[Iterable, np.ndarray, pd.Series]
        :param additional_params: Additional parameters to pass to the Jensen-Shannon Divergence calculation, defaults to None
        :type additional_params: Dict[str, Any], optional
        :return:
        """
        gen_texts = to_iterable(generated_texts)
        ref_texts = to_iterable(reference_texts)

        params = {}
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
