# Visualization

GAICo includes functions to help you visualize comparison results, making it easier to understand model performance across different metrics. These functions are used internally by the [Experiment Class](experiment_class.md) when `plot=True` is set in the `compare()` method, but you can also use them directly for custom plotting needs.

The primary visualization functions are found in the `gaico.visualize` module:

*   `plot_metric_comparison()`: Generates a bar plot comparing different models based on a **single metric**.
*   `plot_radar_comparison()`: Generates a radar (spider) plot comparing multiple models across **several metrics**.

**Prerequisites:** These functions require `matplotlib`, `seaborn`, `numpy`, and `pandas` to be installed. GAICo attempts to import them, and will raise an `ImportError` if they are not available.

## Preparing Data for Plotting

Both plotting functions expect input data as a pandas DataFrame in a "long" format. This DataFrame typically has columns for model names, metric names, and scores. The `gaico.utils.prepare_results_dataframe()` function is designed to convert a nested dictionary of scores into this format.

### `prepare_results_dataframe()`

This utility takes a dictionary where keys are model names and values are dictionaries of metric names to scores (or nested score dictionaries, like those from ROUGE or BERTScore).

```python
from gaico.utils import prepare_results_dataframe
import pandas as pd

# Example raw scores (perhaps from direct metric calculations)
raw_scores_data = {
    'Model_A': {'Jaccard': 0.8, 'ROUGE': {'rouge1': 0.75, 'rougeL': 0.70}, 'Levenshtein': 0.85},
    'Model_B': {'Jaccard': 0.6, 'ROUGE': {'rouge1': 0.65, 'rougeL': 0.60}, 'Levenshtein': 0.70},
    'Model_C': {'Jaccard': 0.9, 'ROUGE': {'rouge1': 0.80, 'rougeL': 0.78}, 'Levenshtein': 0.92}
}

# Convert to a long-format DataFrame
plot_df = prepare_results_dataframe(raw_scores_data)
print("Prepared DataFrame for Plotting:")
print(plot_df)

# Expected Output:
# Prepared DataFrame for Plotting:
#    model_name   metric_name  score
# 0     Model_A       Jaccard   0.80
# 1     Model_A  ROUGE_rouge1   0.75
# 2     Model_A  ROUGE_rougeL   0.70
# 3     Model_A   Levenshtein   0.85
# 4     Model_B       Jaccard   0.60
# 5     Model_B  ROUGE_rouge1   0.65
# 6     Model_B  ROUGE_rougeL   0.60
# 7     Model_B   Levenshtein   0.70
# 8     Model_C       Jaccard   0.90
# 9     Model_C  ROUGE_rouge1   0.80
# 10    Model_C  ROUGE_rougeL   0.78
# 11    Model_C   Levenshtein   0.92
```
This `plot_df` is now ready to be used with the visualization functions.

## 1. Bar Plot: `plot_metric_comparison()`

Use this function to compare models on a single, specific metric.

### Key Parameters:
*   `df` (`pd.DataFrame`): The long-format DataFrame (from `prepare_results_dataframe`).
*   `metric_name` (`str`): **Required.** The name of the metric to plot (e.g., "Jaccard", "ROUGE_rouge1").
*   `aggregate_func` (Optional `Callable`): Aggregation function if multiple scores exist per model for the chosen metric (e.g., `numpy.mean`). Defaults to `numpy.mean`.
*   `model_col`, `score_col`, `metric_col` (Optional `str`): Names of columns for model, score, and metric. Defaults to "model_name", "score", "metric_name".
*   `title`, `xlabel`, `ylabel` (Optional `str`): Plot customization.
*   `figsize` (Optional `tuple`): Figure size.
*   `axis` (Optional `matplotlib.axes.Axes`): Existing Matplotlib Axes to plot on.

### Example:

```python
from gaico.visualize import plot_metric_comparison
import matplotlib.pyplot as plt # For plt.show()
import numpy as np # For np.mean (default aggregate_func)

# Assuming plot_df is available from the previous example

# Plot Jaccard scores
plot_metric_comparison(plot_df, metric_name="Jaccard", title="Jaccard Similarity Comparison")
plt.show()

# Plot ROUGE_rouge1 scores
plot_metric_comparison(plot_df, metric_name="ROUGE_rouge1", title="ROUGE-1 F1 Score Comparison")
plt.show()
```
This will generate and display bar charts, each showing the specified metric's scores for Model_A, Model_B, and Model_C.

## 2. Radar Plot: `plot_radar_comparison()`

Use this function to get a multi-dimensional view of how models perform across several metrics simultaneously. Radar plots are most effective with 3 to 10-12 metrics.

### Key Parameters:
*   `df` (`pd.DataFrame`): The long-format DataFrame.
*   `metrics` (Optional `List[str]`): A list of metric names (e.g., ["Jaccard", "ROUGE_rouge1", "Levenshtein"]) to include in the radar plot. If `None`, all metrics present in the `df` for the models are used.
*   `aggregate_func` (Optional `Callable`): Aggregation function. Defaults to `numpy.mean`.
*   `model_col`, `score_col`, `metric_col` (Optional `str`): Column names.
*   `title` (Optional `str`): Plot title.
*   `figsize` (Optional `tuple`): Figure size.
*   `axis` (Optional `matplotlib.axes.Axes`): Existing Matplotlib polar Axes to plot on.

### Example:

```python
from gaico.visualize import plot_radar_comparison
import matplotlib.pyplot as plt # For plt.show()
import numpy as np # For np.mean

# Assuming plot_df is available

# Define which metrics to include in the radar plot
metrics_for_radar = ["Jaccard", "ROUGE_rougeL", "Levenshtein"]

plot_radar_comparison(plot_df, metrics=metrics_for_radar, title="Overall Model Performance Radar")
plt.show()

# If you want to plot all available metrics (that have scores for the models)
# plot_radar_comparison(plot_df, title="Overall Model Performance Radar (All Metrics)")
# plt.show()
```
This will generate a radar plot where each axis represents one of the `metrics_for_radar`, and each model (Model_A, Model_B, Model_C) is represented by a colored shape connecting its scores on these axes.

By using these visualization tools directly, you can create custom plots tailored to specific analyses or integrate them into larger reporting dashboards. Remember to have the necessary plotting libraries installed in your environment.
