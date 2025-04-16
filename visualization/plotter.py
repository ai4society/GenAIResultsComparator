import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional, Dict, List, Any

def plot_metric_comparison(
    scores_df: pd.DataFrame,
    metric_name: str,
    model_col: str = "model_name",
    score_col: str = "score",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Model",
    ylabel: Optional[str] = None,
    figsize: Optional[tuple] = (10, 6),
    axis: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Generates a bar plot comparing different models based on a single metric.

    :param scores_df: DataFrame containing the scores.
        Expected columns: model_col, 'metric_name', score_col.
    :type scores_df: pd.DataFrame
    :param metric_name: The name of the metric to plot (must exist in the 'metric_name' column).
    :type metric_name: str
    :param model_col: Name of the column identifying the models/generators.
    :type model_col: str
    :param score_col: Name of the column containing the scores.
    :type score_col: str
    :param title: Optional title for the plot. Defaults to f"{metric_name} Comparison".
    :type title: Optional[str]
    :param xlabel: Optional label for the x-axis.
    :type xlabel: Optional[str]
    :param ylabel: Optional label for the y-axis. Defaults to metric_name.
    :type ylabel: Optional[str]
    :param figsize: Figure size for the plot.
    :type figsize: tuple
    :param axis: Optional matplotlib Axes object to plot on. If None, a new figure and axes are created.
    :type axis: Optional[plt.Axes]
    :param kwargs: Additional keyword arguments passed to seaborn.barplot.
    :return: The matplotlib Axes object containing the plot.
    :rtype: plt.Axes
    """
    required_cols = [model_col, "metric_name", score_col]

    metric_data = scores_df[scores_df["metric_name"] == metric_name]

    if axis is None:
        fig, axis = plt.subplots(figsize=figsize)

    sns.barplot(
        data=metric_data, x=model_col, y=score_col, axis=axis, **kwargs
    )

    plot_title = title if title else f"{metric_name} Comparison"
    plot_ylabel = ylabel if ylabel else metric_name

    axis.set_title(plot_title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(plot_ylabel)
    axis.tick_params(axis="x", rotation=45) # Rotate labels if they overlap
    plt.tight_layout() # Adjust layout

    return axis

# Placeholder for future plotting functions

# def plot_score_distribution(...) -> plt.Axes:
#     """Plots the distribution of scores for a metric (e.g., histogram or box plot)."""
#     pass

# def plot_multiple_metrics(...) -> plt.Axes:
#     """Plots multiple metrics across models (e.g., grouped bar chart)."""
#     pass