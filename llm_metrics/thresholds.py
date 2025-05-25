import re
from typing import List, Dict, Any, Union, Optional

from arrow import get
from click import Option

# Default threshold for each metric
DEFAULT_THRESHOLD : Dict[str, float] = {
    "BLEU": 0.5,
    "ROUGE": 0.5,
    "JSD": 0.5,
    "BERTScore": 0.5,
    "Jaccard": 0.5,
    "Cosine": 0.5,
    "Levenshtein": 0.5,
    "SequenceMatcher": 0.5,
} 

def get_default_thresholds() -> Dict[str, float]:
    """
    Returns the default thresholds for each metric.
    This is useful for testing and can be overridden by user-defined thresholds.

    :return: A dictionary of default thresholds for each metric.
        e.g., {"BLEU": 0.5, "JSD": 0.5}
    """
    return DEFAULT_THRESHOLD.copy()

def apply_thresholds(
    results: Union[Dict[str, Union[float, Any]], List[Dict[str, Union[float, Any]]]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Union[Dict[str, Dict[str, Union[float, bool, None]]], List[Dict[str, Dict[str, Union[float, bool, None]]]]]:
    """
    Apply thresholds to scores for single pair or batch of generated and reference texts.

    :param results: Either a single dictionary of scores or a list of score dictionaries
        Single: {"BLEU": 0.6, "JSD": 0.1}
        Batch: [{"BLEU": 0.6, "JSD": 0.1}, {"BLEU": 0.4, "JSD": 0.2}]
    :param thresholds: Dictionary of metric names to threshold values.
        Defaults to get_default_thresholds() if not provided.
    :return: For single input, returns a dictionary. For batch input, returns a list.
        Single: {"BLEU": {"score": 0.6, "threshold_applied": 0.5, "passed_threshold": True}, ...}
        Batch: [{"BLEU": {"score": 0.6, ...}, ...}, {"BLEU": {"score": 0.4, ...}, ...}]
    """
    
    current_threshold = thresholds if thresholds is not None else get_default_thresholds()
    
    def single_result(
        result: Dict[str, Union[float, Any]]
    )-> Dict[str, Dict[str, Union[float, bool, None]]]:
        """
        Apply thresholds to a single result dictionary.
        """
        pair_results = {}
        for metric_name, score in result.items():
            if metric_name in current_threshold:
                threshold_value = current_threshold[metric_name]
                passed = False
                
                if isinstance(score, (int, float)):
                    if metric_name == "JSD":
                        passed = ((1-score) >= threshold_value)
                    else:
                        passed = (score >= threshold_value)
                        
                
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
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Calculate pass/fail percentages for each metric across results.
    
    :param results: Dictionary where keys are metric names and values are lists of scores
    :param thresholds: Dictionary of thresholds for each metric
    :return: Dictionary with metric names as keys and pass/fail statistics as values
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
            "fail_percentage": (failed_count / total_items * 100) if total_items > 0 else 0
        }
    
    return metric_stats