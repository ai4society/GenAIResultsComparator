from collections import Counter
from typing import Any, Dict, List, Union, Optional

import numpy as np
import pandas as pd
import os


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
                    if isinstance(
                        sub_score, (int, float)
                    ):  # Ensure the final score is numeric
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


def generate_deltas_csv(
    threshold_results: Union[
        Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]
    ],
    generated_texts: Optional[Union[str, List[str]]] = None,
    reference_texts: Optional[Union[str, List[str]]] = None,
    output_csv_path: Optional[str] = None,
) -> None:
    """
    Generate a CSV file from threshold function outputs with optional text strings.

    :param threshold_results: Output from apply_thresholds (handles both single and batch)
        Single: {"BLEU": {"score": 0.6, "threshold_applied": 0.5, "passed_threshold": True}, ...}
        Batch: [{"BLEU": {"score": 0.6, ...}, ...}, {"BLEU": {"score": 0.4, ...}, ...}]
    :param generated_texts: Optional generated text string(s)
    :param reference_texts: Optional reference text string(s)
    :param output_csv_path: Optional path to save the CSV file
    """

    if output_csv_path is None:
        output_csv_path = os.path.join(os.getcwd(), "threshold_analysis_report.csv")
        print(f"Output CSV path not provided. Defaulting to: {output_csv_path}")

    # Normalize generated and reference texts to lists
    gen_texts_list: Optional[List[str]] = None
    ref_texts_list: Optional[List[str]] = None
    results_list: Optional[List[Dict[str, Any]]] = None

    if isinstance(threshold_results, dict):
        # Single pair
        results_list = [threshold_results]
        if generated_texts is not None:
            gen_texts_list = (
                [generated_texts]
                if isinstance(generated_texts, str)
                else generated_texts
            )
        if reference_texts is not None:
            ref_texts_list = (
                [reference_texts]
                if isinstance(reference_texts, str)
                else reference_texts
            )
    else:
        # Already a list
        results_list = threshold_results
        if isinstance(generated_texts, list):
            gen_texts_list = generated_texts
        if isinstance(reference_texts, list):
            ref_texts_list = reference_texts

    report_data: List[Dict[str, Any]] = []

    for idx, item_results in enumerate(results_list):
        row_data: Dict[str, Any] = {}

        # Adding text strings if provided
        if gen_texts_list is not None and idx < len(gen_texts_list):
            row_data["generated_text"] = str(gen_texts_list[idx])
        if ref_texts_list is not None and idx < len(ref_texts_list):
            row_data["reference_text"] = str(ref_texts_list[idx])

        # Adding metric scores and pass/fail status
        for metric_name, details in item_results.items():
            row_data[f"{metric_name}_score"] = details.get("score")
            row_data[f"{metric_name}_passed"] = details.get("passed_threshold")

        report_data.append(row_data)

    if not report_data:
        print("Warning: No data to write to CSV.")
        return

    # Creating DataFrame
    df = pd.DataFrame(report_data)

    # Create output directory for the CSV file if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_csv_path, index=False)
    print(f"CSV report generated at: {output_csv_path}")
