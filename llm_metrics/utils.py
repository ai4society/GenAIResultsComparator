from collections import Counter
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


def to_iterable(obj: Any) -> Union[np.ndarray, pd.Series, List]:
    """
    Convert object to an iterable, preserving numpy arrays and pandas Series.

    :param obj: The object to convert
    :type obj: Any
    :return: An iterable version of the object
    :rtype: Union[np.ndarray, pd.Series, List]
    """
    if isinstance(obj, (np.ndarray, pd.Series)):
        return obj
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return list(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.values
    elif isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, str):
        return [obj]
    else:
        try:
            return list(iter(obj))
        except TypeError:
            return [obj]


def get_ngrams(text: str, n: int) -> Dict[str, int]:
    """
    Generate n-grams from a given text.

    :param text: The input text
    :type text: str
    :param n: The number of words in each n-gram
    :type n: int
    :return: A dictionary of n-grams and their counts
    :rtype: Dict[str, int]
    """
    words: List[str] = text.lower().split()  # Split the text into words
    ngrams: zip = zip(*[words[i:] for i in range(n)])  # Create n-grams
    return Counter(" ".join(ngram) for ngram in ngrams)  # Count the n-grams


def batch_get_ngrams(
    texts: Union[np.ndarray, pd.Series, List[str]], n: int
) -> List[Dict[str, int]]:
    """
    Generate n-grams for a batch of texts.

    :param texts: The input texts
    :type texts: Union[np.ndarray, pd.Series, List[str]]
    :param n: The number of words in each n-gram
    :type n: int
    :return: A list of dictionaries of n-grams and their counts
    :rtype: List[Dict[str, int]]
    """
    if isinstance(texts, np.ndarray):
        return [get_ngrams(text, n) for text in texts]
    elif isinstance(texts, pd.Series):
        return texts.apply(lambda x: get_ngrams(x, n)).tolist()
    else:
        return [get_ngrams(text, n) for text in texts]


def prepare_results_dataframe(
    results_dict: Dict[str, Dict[str, Any]],
    model_col: str = "model_name",
    metric_col: str = "metric_name",
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Converts a nested dictionary of results into a long-format DataFrame suitable for plotting.

    Example Input `results_dict`:
    {
        'ModelA': {'BLEU': 0.8, 'ROUGE': {'f1': 0.75}},
        'ModelB': {'BLEU': 0.7, 'ROUGE': {'f1': 0.65}}
    }
    Example Output DataFrame:
       model_name  metric_name  score
    0     ModelA    BLEU       0.80
    1     ModelA    ROUGE_f1   0.75
    2     ModelB    BLEU       0.70
    3     ModelB    ROUGE_f1   0.65

    :param results_dict: Nested dictionary where keys are model names and values are dictionaries of metric names to scores (or nested score dicts).
    :type results_dict: Dict[str, Dict[str, Any]]
    :param model_col: Name for the column containing model names in the output DataFrame.
    :type model_col: str
    :param metric_col: Name for the column containing metric names in the output DataFrame.
    :type metric_col: str
    :param score_col: Name for the column containing scores in the output DataFrame.
    :type score_col: str
    :return: A pandas DataFrame in long format.
    :rtype: pd.DataFrame
    """

    records = []
    for model_name, metrics_data in results_dict.items():
        for metric_name, score_value in metrics_data.items():
            if isinstance(score_value, dict):
                for sub_metric, sub_score in score_value.items():
                    full_metric_name = f"{metric_name}_{sub_metric}"
                    if isinstance(sub_score, (int, float)):  # Ensure the final score is numeric
                        records.append(
                            {
                                model_col: model_name,
                                metric_col: full_metric_name,
                                score_col: sub_score,
                            }
                        )
                    # Handle cases with deeper nesting or other types if needed
            elif isinstance(score_value, (int, float)):
                records.append(
                    {
                        model_col: model_name,
                        metric_col: metric_name,
                        score_col: score_value,
                    }
                )
            # Handle for other types if necessary (e.g., lists of scores)

    if not records:
        return pd.DataFrame(columns=[model_col, metric_col, score_col])

    return pd.DataFrame(records)
