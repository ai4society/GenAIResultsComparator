from typing import Any, Dict, List, Optional, TypeAlias

# Default threshold for each metric
DEFAULT_THRESHOLD: Dict[str, float] = {
    "BLEU": 0.5,
    "ROUGE": 0.5,
    "JSD": 0.5,
    "BERTScore": 0.5,
    "Jaccard": 0.5,
    "Cosine": 0.5,
    "Levenshtein": 0.5,
    "SequenceMatcher": 0.5,
}

# Type alias for results with scores and thresholds
ThresholdedResults: TypeAlias = Dict[str, float | Any]


def get_default_thresholds() -> Dict[str, float]:
    """
    Returns the default thresholds for each metric.
    This is useful for testing and can be overridden by user-defined thresholds.

    :return: A dictionary of default thresholds for each metric.
        e.g., {"BLEU": 0.5, "JSD": 0.5}
    :rtype: Dict[str, float]
    """
    return DEFAULT_THRESHOLD.copy()


def apply_thresholds(
    results: ThresholdedResults | List[ThresholdedResults],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, ThresholdedResults] | List[Dict[str, ThresholdedResults]]:
    """
    Apply thresholds to scores for single pair or batch of generated and reference texts.
    Type ThresholdedResults is a dictionary where keys are metric names and values are either scores or dictionaries with scores.
    Specifically, it is of type Dict[str, float | Any].

    :param results: Either a single dictionary of scores or a list of score dictionaries
        Single: {"BLEU": 0.6, "JSD": 0.1}
        Batch: [{"BLEU": 0.6, "JSD": 0.1}, {"BLEU": 0.4, "JSD": 0.2}]
    :rtype results: ThresholdedResults | List[ThresholdedResults]
    :param thresholds: Dictionary of metric names to threshold values.
        Defaults to get_default_thresholds() if not provided.
    :return: For single input, returns a dictionary. For batch input, returns a list.
        Single: {"BLEU": {"score": 0.6, "threshold_applied": 0.5, "passed_threshold": True}, ...}
        Batch: [{"BLEU": {"score": 0.6, ...}, ...}, {"BLEU": {"score": 0.4, ...}, ...}]
    """

    current_threshold = thresholds if thresholds is not None else get_default_thresholds()

    def single_result(
        result: ThresholdedResults,
    ) -> Dict[str, ThresholdedResults]:
        """
        Apply thresholds to a single result dictionary.

        Type ThresholdedResults is a dictionary where keys are metric names and values are either scores or dictionaries with scores.
        Specifically, it is of type Dict[str, float | Any].

        :param result: A dictionary of scores for a single pair of generated and reference texts.
        :type result: ThresholdedResults
        :return: A dictionary where each metric's score is accompanied by its threshold and pass/fail status.
        :rtype: Dict[str, ThresholdedResults]
        """
        pair_results = {}
        for metric_name, score in result.items():
            if metric_name in current_threshold:
                threshold_value = current_threshold[metric_name]
                passed = False

                if isinstance(score, (int, float)):
                    if metric_name == "JSD":
                        passed = (1 - score) >= threshold_value
                    else:
                        passed = score >= threshold_value

                pair_results[metric_name] = {
                    "score": score,
                    "threshold_applied": threshold_value,
                    "passed_threshold": passed,
                }
        return pair_results

    if isinstance(results, dict):
        return single_result(results)
    elif isinstance(results, list):
        return [single_result(r) for r in results]
    else:
        raise TypeError(f"Expected dict or list of dicts, got {type(results)}")


def calculate_pass_fail_percent(
    results: Dict[str, List[float]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float | int]]:
    """
    Calculate pass/fail percentages for each metric across results.

    :param results: Dictionary where keys are metric names and values are lists of scores
    :type results: Dict[str, List[float]]
    :param thresholds: Dictionary of thresholds for each metric
    :type thresholds: Optional[Dict[str, float]]
    :return: Dictionary with metric names as keys and pass/fail statistics as values
    :rtype: Dict[str, Dict[str, float | int]]
    """
    current_thresholds = thresholds if thresholds is not None else get_default_thresholds()

    if not results:
        return {}

    metric_stats = {}

    for metric_name, scores_list in results.items():
        if metric_name not in current_thresholds:
            continue

        threshold = current_thresholds[metric_name]
        total_items = len(scores_list)
        passed_count = 0

        for score in scores_list:
            if isinstance(score, (int, float)):
                if metric_name == "JSD":
                    if (1 - score) >= threshold:
                        passed_count += 1
                else:
                    if score >= threshold:
                        passed_count += 1

        failed_count = total_items - passed_count

        metric_stats[metric_name] = {
            "total_passed": passed_count,
            "total_failed": failed_count,
            "pass_percentage": (passed_count / total_items * 100) if total_items > 0 else 0,
            "fail_percentage": (failed_count / total_items * 100) if total_items > 0 else 0,
        }

    return metric_stats
