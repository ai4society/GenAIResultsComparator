# Working with Thresholds

GAICo provides utilities to apply thresholds to metric scores and analyze pass/fail statistics. This is useful for determining if generated text meets a certain quality bar for specific metrics. While the [Experiment Class](experiment_class.md) can apply thresholds automatically, you can also use these functions directly.

## Key Threshold Functions

The primary functions for working with thresholds are located in the `gaico.thresholds` module:

*   `get_default_thresholds()`: Returns a dictionary of the library's default threshold values for each base metric.
*   `apply_thresholds()`: Applies specified or default thresholds to a dictionary (or list of dictionaries) of metric scores.
*   `calculate_pass_fail_percent()`: Calculates pass/fail percentages for a collection of scores for each metric.

## 1. Getting Default Thresholds

You can inspect the default thresholds used by the library:

```python
from gaico.thresholds import get_default_thresholds

default_thresholds = get_default_thresholds()
print("Default Thresholds:")
for metric, threshold_val in default_thresholds.items():
    print(f"- {metric}: {threshold_val}")

# Example Output:
# Default Thresholds:
# - BLEU: 0.5
# - ROUGE: 0.5
# - JSD: 0.5
# - BERTScore: 0.5
# - Jaccard: 0.5
# - Cosine: 0.5
# - Levenshtein: 0.5
# - SequenceMatcher: 0.5
```
The `DEFAULT_THRESHOLD` constant in `gaico.thresholds` also holds these values.

## 2. Applying Thresholds with `apply_thresholds()`

This function takes your calculated metric scores and a dictionary of thresholds, then returns a detailed structure indicating the score, the threshold applied, and whether the score passed the threshold.

### Input:
*   `results` (`Dict[str, Union[float, Any]] | List[Dict[str, Union[float, Any]]]`):
    *   For a single evaluation: A dictionary where keys are metric names (e.g., "Jaccard", "ROUGE_rouge1") and values are their scores.
    *   For batch evaluations: A list of such dictionaries.
*   `thresholds` (Optional `Dict[str, float]`):
    *   A dictionary mapping metric names to their desired threshold values.
    *   If `None`, `get_default_thresholds()` is used.
    *   **Note:** For JSDivergence, "passing" means `(1 - score) >= threshold_value` because lower JSD is better. For other metrics, "passing" means `score >= threshold_value`.

### Output:
A dictionary (or list of dictionaries) where each metric entry contains:
*   `"score"`: The original score.
*   `"threshold_applied"`: The threshold value used for this metric.
*   `"passed_threshold"`: A boolean indicating if the score passed.

### Example:

```python
from gaico.thresholds import apply_thresholds, get_default_thresholds

# Example scores for a single evaluation
single_eval_scores = {
    "Jaccard": 0.75,
    "ROUGE_rouge1": 0.45,
    "Levenshtein": 0.80,
    "JSD": 0.2 # Lower JSD is better
}

# Using default thresholds
thresholded_results_default = apply_thresholds(single_eval_scores)
print("Thresholded Results (Default):")
for metric, details in thresholded_results_default.items():
    print(f"  {metric}: Score={details['score']}, Threshold={details['threshold_applied']}, Passed={details['passed_threshold']}")

# Thresholded Results (Default):
#   Jaccard: Score=0.75, Threshold=0.5, Passed=True
#   Levenshtein: Score=0.8, Threshold=0.5, Passed=True
#   JSD: Score=0.2, Threshold=0.5, Passed=True

# Using custom thresholds
custom_thresholds = {
    "Jaccard": 0.7,
    "ROUGE_rouge1": 0.5,
    "Levenshtein": 0.75,
    "JSD": 0.6 # This means (1 - JSD_score) should be >= 0.6, so JSD_score <= 0.4
}
thresholded_results_custom = apply_thresholds(single_eval_scores, thresholds=custom_thresholds)
print("\nThresholded Results (Custom):")
for metric, details in thresholded_results_custom.items():
    print(f"  {metric}: Score={details['score']}, Threshold={details['threshold_applied']}, Passed={details['passed_threshold']}")

# Thresholded Results (Custom):
#   Jaccard: Score=0.75, Threshold=0.7, Passed=True
#   ROUGE_rouge1: Score=0.45, Threshold=0.5, Passed=False
#   Levenshtein: Score=0.8, Threshold=0.75, Passed=True
#   JSD: Score=0.2, Threshold=0.6, Passed=True

# Example for batch results
batch_eval_scores = [
    {"Jaccard": 0.8, "ROUGE_rouge1": 0.6},
    {"Jaccard": 0.4, "ROUGE_rouge1": 0.3}
]
thresholded_batch = apply_thresholds(batch_eval_scores, thresholds={"Jaccard": 0.5, "ROUGE_rouge1": 0.55})
print("\nThresholded Batch Results (Custom):")
for i, item_results in enumerate(thresholded_batch):
    print(f"  Item {i+1}:")
    for metric, details in item_results.items():
        print(f"    {metric}: Score={details['score']}, Threshold={details['threshold_applied']}, Passed={details['passed_threshold']}")

# Thresholded Batch Results (Custom):
#   Item 1:
#     Jaccard: Score=0.8, Threshold=0.5, Passed=True
#     ROUGE_rouge1: Score=0.6, Threshold=0.55, Passed=True
#   Item 2:
#     Jaccard: Score=0.4, Threshold=0.5, Passed=False
#     ROUGE_rouge1: Score=0.3, Threshold=0.55, Passed=False
```

The output from `apply_thresholds` is also what `gaico.utils.generate_deltas_frame` (used by `Experiment.compare()` for CSV output) expects.

## 3. Calculating Pass/Fail Percentages with `calculate_pass_fail_percent()`

If you have a collection of scores for multiple items (e.g., from evaluating many generated texts against their references) and want to see overall pass/fail rates for each metric, this function is useful.

### Input:
*   `results` (`Dict[str, List[float]]`):
    A dictionary where keys are metric names and values are lists of scores obtained for that metric across multiple evaluations.
*   `thresholds` (Optional `Dict[str, float]`):
    Custom thresholds to use. Defaults to `get_default_thresholds()`.

### Output:
A dictionary where keys are metric names. Each value is another dictionary containing:
*   `"total_passed"`: Count of items that passed the threshold.
*   `"total_failed"`: Count of items that failed.
*   `"pass_percentage"`: Percentage of items that passed.
*   `"fail_percentage"`: Percentage of items that failed.

### Example:

```python
from gaico.thresholds import calculate_pass_fail_percent

# Example: Scores from multiple evaluations for different metrics
batch_scores_for_stats = {
    "Jaccard": [0.8, 0.4, 0.9, 0.6, 0.7],
    "Levenshtein": [0.9, 0.85, 0.6, 0.77, 0.92],
    "JSD": [0.1, 0.5, 0.05, 0.6, 0.2] # Lower is better
}

custom_thresholds_for_stats = {
    "Jaccard": 0.7,
    "Levenshtein": 0.8,
    "JSD": 0.6 # (1 - JSD_score) >= 0.6  => JSD_score <= 0.4
}

pass_fail_stats = calculate_pass_fail_percent(batch_scores_for_stats, thresholds=custom_thresholds_for_stats)

print("\nPass/Fail Statistics:")
for metric, stats in pass_fail_stats.items():
    print(f"  Metric: {metric}")
    print(f"    Total Passed: {stats['total_passed']}")
    print(f"    Total Failed: {stats['total_failed']}")
    print(f"    Pass Percentage: {stats['pass_percentage']:.2f}%")
    print(f"    Fail Percentage: {stats['fail_percentage']:.2f}%")

# Pass/Fail Statistics:
#   Metric: Jaccard
#     Total Passed: 3
#     Total Failed: 2
#     Pass Percentage: 60.00%
#     Fail Percentage: 40.00%
#   Metric: Levenshtein
#     Total Passed: 3
#     Total Failed: 2
#     Pass Percentage: 60.00%
#     Fail Percentage: 40.00%
#   Metric: JSD
#     Total Passed: 3
#     Total Failed: 2
#     Pass Percentage: 60.00%
#     Fail Percentage: 40.00%
```

These thresholding utilities provide flexible ways to interpret your metric scores beyond just their raw values, helping you make more informed decisions about model performance.
