from typing import Any, Callable, Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from .base import BaseMetric

# Conditional imports for NLTK (BLEU, JSDivergence)
_nltk_available = False
_SmoothingFunction_cls = None
_corpus_bleu_func = None
_sentence_bleu_func = None
_FreqDist_cls = None
try:
    # import nltk  # Check if nltk itself is available
    from nltk import FreqDist as _ImportedFreqDist
    from nltk.translate.bleu_score import SmoothingFunction as _ImportedSmoothingFunction
    from nltk.translate.bleu_score import corpus_bleu as _ImportedCorpusBleu
    from nltk.translate.bleu_score import sentence_bleu as _ImportedSentenceBleu

    _SmoothingFunction_cls = _ImportedSmoothingFunction
    _corpus_bleu_func = _ImportedCorpusBleu
    _sentence_bleu_func = _ImportedSentenceBleu
    _FreqDist_cls = _ImportedFreqDist
    _nltk_available = True
except ImportError:
    pass  # Handled in class __init__

# Conditional imports for SciPy (JSDivergence)
_scipy_available = False
_jensenshannon_func = None
try:
    from scipy.spatial.distance import jensenshannon as _ImportedJensenShannon

    _jensenshannon_func = _ImportedJensenShannon
    _scipy_available = True
except ImportError:
    pass  # Handled in class __init__

__nltk_available__ = _nltk_available
__scipy_available__ = _scipy_available


class BLEU(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) score implementation.
    This class provides methods to calculate BLEU scores for individual sentence pairs and for batches of sentences.
    It uses the NLTK library to calculate BLEU scores.
    """

    def __init__(
        self,
        n: int = 4,
        smoothing_function: Optional[Callable] = None,
    ):
        """
        Initialize the BLEU scorer with the specified parameters.

        :param n: The max n-gram order to use for BLEU calculation, defaults to 4
        :type n: int
        :param smoothing_function: The smoothing function to use for BLEU, defaults to SmoothingFunction.method1 from NLTK
        :type smoothing_function: Optional[Callable]
        """
        if not _nltk_available:
            raise ImportError("NLTK is not installed, which is required for BLEU metric. ")
        self.n = n
        # _SmoothingFunction_cls will be None if nltk failed to import
        self.smoothing_function = (
            smoothing_function
            if smoothing_function is not None
            else _SmoothingFunction_cls().method1  # type: ignore
        )

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any,
    ) -> float:
        """
        Calculate the BLEU score for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional parameters to pass to the NLTK sentence_bleu function
        :type kwargs: Any
        :return: The BLEU score for the text pair
        :rtype: float
        """

        # Split texts into words and calculate sentence-level BLEU score
        score = _sentence_bleu_func(  # type: ignore
            [reference_text.split()],
            generated_text.split(),
            smoothing_function=self.smoothing_function,  # type: ignore
            **kwargs,
        )
        score = cast(float, score)  # Signal to mypy that score is a float
        return float(score)  # Ensure the score is returned as a float

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        use_corpus_bleu: bool = True,
        **kwargs: Any,
    ) -> float | List[float] | np.ndarray | pd.Series:
        """
        Calculate BLEU scores for a batch of generated and reference texts.

        This method supports two modes of calculation:
        1. Corpus-level BLEU (default):
            Calculates a single BLEU score for the entire corpus. Uses the `corpus_bleu` method from NLTK
        2. Sentence-level BLEU:
            Calculates individual BLEU scores for each sentence pair. Uses the `sentence_bleu` method from NLTK

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param use_corpus_bleu: Whether to use corpus-level BLEU calculation, defaults to True
        :type use_corpus_bleu: bool
        :param kwargs: Additional parameters to pass to NLTK BLEU functions
        :type kwargs: Any
        :return: Either a single corpus-level BLEU score or a list/array/series of sentence-level BLEU scores.
        :rtype: float | List[float] | np.ndarray | pd.Series
        """

        if use_corpus_bleu:
            # Prepare data for corpus_bleu calculation
            if isinstance(generated_texts, (np.ndarray, pd.Series)):
                # For numpy arrays and pandas Series, use vectorized operations
                hypotheses = [text.split() for text in generated_texts]
                references = [[text.split()] for text in reference_texts]
            else:
                # For other iterable types, we use a list comprehension
                hypotheses = [gen.split() for gen in generated_texts]
                references = [[ref.split()] for ref in reference_texts]

            # Calculate and return the corpus-level BLEU score
            score = _corpus_bleu_func(  # type: ignore
                references,
                hypotheses,
                smoothing_function=self.smoothing_function,  # type: ignore
                **kwargs,
            )
            score = cast(float, score)  # Signal to mypy that score is a float
            return float(score)  # Ensure the score is returned as a float
        else:
            # Calculate individual BLEU scores for each sentence pair
            if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
                # Use numpy vectorization for faster calculation
                return np.array(
                    [
                        self._single_calculate(gen, ref, **kwargs)
                        for gen, ref in zip(generated_texts, reference_texts)
                    ]
                )
            elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
                # Use pandas' apply method for Series
                return generated_texts.combine(
                    reference_texts,
                    lambda g, r: self._single_calculate(g, r, **kwargs),
                )
            else:
                # For other iterable types, use a list comprehension
                return [
                    self._single_calculate(gen, ref, **kwargs)
                    for gen, ref in zip(generated_texts, reference_texts)
                ]


class ROUGE(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score implementation using the `rouge_score` library.
    """

    def __init__(
        self,
        rouge_types: Optional[List[str]] = None,
        use_stemmer: bool = True,
        **kwargs: Any,
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
        :param kwargs: Additional parameters to pass to the ROUGE calculation, defaults to None
            Default only passes the `use_stemmer` parameter
        :type kwargs: Any
        """
        self.params = {"use_stemmer": use_stemmer}
        self.params.update(kwargs)

        # Check if rouge_types is valid
        if rouge_types:
            if not isinstance(rouge_types, list):
                raise ValueError("rouge_types must be a list")
            elif not all(val in ["rouge1", "rouge2", "rougeL"] for val in rouge_types):
                raise ValueError("rouge_types must be one of ['rouge1', 'rouge2', 'rougeL']")

        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]

        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, **self.params)

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any,
    ) -> Dict[str, float] | float:
        """
        Calculate the ROUGE score for a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional parameters for the ROUGE scorer, defaults to None
        :type kwargs: Any
        :return: Either a single score or a dictionary of scores containing ROUGE types
        :rtype: Union[dict, float]
        """
        self.params.update(kwargs)

        # For some texts, such as empty strings, the scorer may return 0
        # To account for such cases, ensure that each score is converted to float
        scores = self.scorer.score(reference_text, generated_text)
        score_dict = {k: float(v.fmeasure) for k, v in scores.items()}

        # Return based on the supplied ROUGE types
        if len(self.rouge_types) == 1:
            return float(score_dict.get(self.rouge_types[0], 0.0))
        return {key: float(score_dict[key]) for key in self.rouge_types}

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> list[float] | list[dict] | np.ndarray | pd.Series:
        """
        Calculate ROUGE scores for a batch of generated and reference texts.
        Supports iterables, numpy arrays, and pandas Series as input and output.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional parameters for the ROUGE scorer, defaults to None
        :type kwargs: Any
        :return: A list, numpy array, or pandas Series of Dictionary of ROUGE scores
        :rtype: list[dict] | np.ndarray | pd.Series
        """
        # `self.scorer` takes care of calculating ROUGE scores based on the supplied ROUGE types
        scores: list[Dict[str, float]] = [
            cast(Dict[str, float], self._single_calculate(gen, ref, **kwargs))
            for gen, ref in zip(generated_texts, reference_texts)
        ]

        if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
            return np.array(scores)

        elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
            return pd.Series(scores, index=generated_texts.index)

        else:
            return scores


class JSDivergence(BaseMetric):
    """Jensen-Shannon Divergence metric implementation using the `scipy` library."""

    def __init__(
        self,
    ):
        """Initialize the Jensen-Shannon Divergence metric."""
        if not _nltk_available or not _scipy_available:
            missing_deps = []
            if not _nltk_available:
                missing_deps.append("nltk")
            if not _scipy_available:
                missing_deps.append("scipy")
            raise ImportError(
                f"{', '.join(missing_deps)} is not installed, which is required for JSDivergence metric. "
                "Please install it with: pip install 'gaico[jsd]'"
            )

    def _single_calculate(
        self,
        generated_text: str,
        reference_text: str,
        **kwargs: Any,
    ) -> float:
        """
        Calculate the Jensen-Shannon Divergence between a pair of generated and reference texts.

        :param generated_text: The generated text to evaluate
        :type generated_text: str
        :param reference_text: The reference text to compare against
        :type reference_text: str
        :param kwargs: Additional parameters for scipy's jensenshannon function
        :type kwargs: Any
        :return: The Jensen-Shannon Divergence score for the text pair
        :rtype: float
        """

        gen_freq = _FreqDist_cls(generated_text.split())  # type: ignore
        ref_freq = _FreqDist_cls(reference_text.split())  # type: ignore
        all_words = set(gen_freq.keys()) | set(ref_freq.keys())

        gen_probs = [gen_freq.freq(word) for word in all_words]
        ref_probs = [ref_freq.freq(word) for word in all_words]

        # Handle the case where arrays are zeros
        if all(x == 0 for x in ref_probs) or all(x == 0 for x in gen_probs):
            return 0.0

        return 1.0 - float((_jensenshannon_func(gen_probs, ref_probs, **kwargs)))  # type: ignore

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> np.ndarray | pd.Series | list[float]:
        """
        Calculate Jensen-Shannon Divergence scores for a batch of generated and reference texts.
        Supports iterables, numpy arrays, and pandas Series as input and output.

        :param generated_texts: Generated texts
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Reference texts
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional parameters for scipy's jensenshannon function, defaults to None
        :type kwargs: Any
        :return: A list, array, or Series of JSD scores
        :rtype: np.ndarray | pd.Series | list[float]
        """

        if isinstance(generated_texts, np.ndarray) and isinstance(reference_texts, np.ndarray):
            return np.array(
                [
                    self._single_calculate(gen, ref, **kwargs)
                    for gen, ref in zip(generated_texts, reference_texts)
                ]
            )

        elif isinstance(generated_texts, pd.Series) and isinstance(reference_texts, pd.Series):
            return generated_texts.combine(
                reference_texts, lambda g, r: self._single_calculate(g, r, **kwargs)
            )

        else:
            return [
                self._single_calculate(gen, ref, **kwargs)
                for gen, ref in zip(generated_texts, reference_texts)
            ]
