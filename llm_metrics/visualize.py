from typing import Optional, Dict, List, Any, Union 

# Optional Imports - Keep these for lazy loading
try:
    import matplotlib.pyplot as plt
    from math import pi
except ImportError:
    # Allow module import, functions will raise runtime error if called without matplotlib/math
    plt = None
    pi = None

try:
    import numpy as np
except ImportError:
    # Allow module import, functions will raise runtime error if called without numpy
    np = None

try:
    import pandas as pd
except ImportError:
    # Allow module import, functions will raise runtime error if called without pandas
    pd = None

try:
    import seaborn as sns
except ImportError:
    # Allow module import, functions will raise runtime error if called without seaborn
    sns = None

# Bar Plot Function for comparing single metric across models 
def plot_metric_comparison(
    scores_df: Any, 
    metric_name: str,
    model_col: str = "model_name",
    score_col: str = "score",
    title: Optional[str] = None,
    xlabel: Optional[str] = "Model",
    ylabel: Optional[str] = None,
    figsize: Optional[tuple] = (10, 6),
    axis: Optional[Any] = None, 
    **kwargs: Any,
) -> Any: 
    """
    Generates a bar plot comparing different models based on a single metric.
    Assumes matplotlib, seaborn, and pandas are installed and available at runtime.

    :param scores_df: DataFrame containing the scores.
        Expected columns: model_col, 'metric_name', score_col.
    :type scores_df: pd.DataFrame or Any
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
    :param figsize: Figure size for the plot if creating a new figure.
    :type figsize: tuple
    :param axis: Optional matplotlib Axes object to plot on. If None, a new figure and axes are created.
    :type axis: Optional[matplotlib.axes.Axes or Any]
    :param kwargs: Additional keyword arguments passed to seaborn.barplot.
    :raises ImportError: If required libraries (matplotlib, seaborn, pandas) are not installed when called.
    :return: The matplotlib Axes object containing the plot.
    :rtype: matplotlib.axes.Axes or Any
    """

    metric_data = scores_df[scores_df["metric_name"] == metric_name]

    # Create plot if axis not provided 
    if axis is None:
        if plt is None:
            raise ImportError("Matplotlib is required but not installed.")
        fig, axis = plt.subplots(figsize=figsize)

    # Generate bar plot 
    if sns is None:
        raise ImportError("Seaborn is required but not installed.")
    sns.barplot(
        data=metric_data, x=model_col, y=score_col, ax=axis, **kwargs
    )

    # Set plot labels and title
    plot_title = title if title else f"{metric_name} Comparison"
    plot_ylabel = ylabel if ylabel else metric_name

    axis.set_title(plot_title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(plot_ylabel)

    # Adjust x-axis tick labels based on number of models
    if len(metric_data[model_col].unique()) > 5:
        axis.tick_params(axis="x", rotation=45)
    else:
        axis.tick_params(axis="x", rotation=0)

    # Adjust layout 
    if plt is None:
        raise ImportError("Matplotlib is required but not installed.")
    plt.tight_layout()

    return axis

# --- Radar Plot Function (Error checks removed as requested) ---
def plot_radar_comparison(
    df: Any, 
    metrics: List[str],
    model_col: str = "model_name",
    score_col: str = "score",
    metric_name_col: str = "metric_name",
    title: Optional[str] = "Model Comparison Radar Plot",
    figsize: Optional[tuple] = (8, 8),
    fill_alpha: float = 0.1,
    line_width: float = 1.0,
    y_ticks: Optional[List[float]] = None,
    axis: Optional[Any] = None, 
    **kwargs: Any,
) -> Any: 
    """
    Generates a radar plot comparing multiple models across several metrics.
    Assumes matplotlib, numpy, and pandas are installed and available at runtime.

    :param df: DataFrame containing the scores in long format.
            Expected columns: model_col, metric_name_col, score_col.
    :type df: pd.DataFrame or Any
    :param metrics: List of metric names (values from metric_name_col) to include in the radar plot axes.
    :type metrics: List[str]
    :param model_col: Name of the column identifying the models.
    :type model_col: str
    :param score_col: Name of the column containing the scores.
    :type score_col: str
    :param metric_name_col: Name of the column containing the metric names.
    :type metric_name_col: str
    :param title: Optional title for the plot.
    :type title: Optional[str]
    :param figsize: Figure size if creating a new plot.
    :type figsize: Optional[tuple]
    :param fill_alpha: Alpha transparency for the filled area under the lines.
    :type fill_alpha: float
    :param line_width: Width of the lines on the plot.
    :type line_width: float
    :param y_ticks: Optional list of values for y-axis ticks (score range). If None, defaults based on data.
    :type y_ticks: Optional[List[float]]
    :param axis: Optional matplotlib Axes object (must be polar). If None, a new figure and polar axes are created.
    :type axis: Optional[matplotlib.axes.Axes or Any]
    :param kwargs: Additional keyword arguments (currently unused but kept for future customization).
    :raises ImportError: If required libraries (matplotlib, numpy, pandas) are not installed when called.
    :raises KeyError: If expected columns are missing in df.
    :raises ValueError: If pivoting fails (e.g., duplicate model/metric pairs) or provided axis is not polar.
    :raises TypeError: If axis is not a valid matplotlib Axes object or other type mismatches occur.
    :return: The matplotlib Axes object containing the plot.
    :rtype: matplotlib.axes.Axes or Any
    """
    # Check for pandas dependency implicitly needed for pivot/reindex
    if pd is None:
        raise ImportError("Pandas is required but not installed.")

    # Filter data 
    plot_data = df[df[metric_name_col].isin(metrics)]

    # Pivot data - raises ValueError on duplicates, KeyError if columns missing
    pivot_df = plot_data.pivot(
        index=model_col, columns=metric_name_col, values=score_col
    )

    pivot_df = pivot_df.reindex(columns=metrics)

    models = pivot_df.index.tolist()
    num_vars = len(metrics)

    # Calculate angles 
    if pi is None:
        raise ImportError("math.pi is required but not available.")
    if num_vars == 0:
        raise ValueError("Metrics list cannot be empty.")
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] # Close the plot

    # Create plot if axis not provided 
    if axis is None:
        if plt is None:
            raise ImportError("Matplotlib is required but not installed.")
        fig, axis = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    elif not hasattr(axis, 'set_theta_offset'):
        raise ValueError("Provided axis must be a polar projection.")

    # Set up plot axes
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(metrics)

    # Determine Y-axis limits and ticks 
    if np is None:
        raise ImportError("Numpy is required but not installed.")
    if y_ticks:
        axis.set_yticks(y_ticks)
    else:
        max_val = pivot_df.max().max()
        if pd.notna(max_val) and max_val > 0:
            # Calculate upper limit and step for ticks
            upper_lim = np.ceil(max_val * 10) / 10
            step = max(0.1, np.round(upper_lim / 5, 1)) # Adjust step logic as needed
            # Ensure step is not zero if upper_lim is very small
            step = step if step > 0 else 0.1
            axis.set_yticks(np.arange(0, upper_lim + step, step=step))
        else:
            axis.set_yticks(np.arange(0, 1.1, 0.2)) # Default ticks

    # Plot data for each model 
    if np is None:
        raise ImportError("Numpy is required but not installed.")
    angles_np = np.array(angles) # Convert angles to numpy array once

    for model in models:
        values = pivot_df.loc[model].values.flatten().tolist()
        values_closed = values + values[:1] # Close the plot
        # Convert to float array to handle potential NaNs for plotting
        values_masked = np.array(values_closed, dtype=float)

        # Plot line segments and fill area
        axis.plot(angles_np, values_masked, linewidth=line_width, linestyle='solid', label=model)
        axis.fill(angles_np, values_masked, alpha=fill_alpha)

    # Add legend and title
    axis.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)) # Adjust legend position as needed
    if title:
        axis.set_title(title, size=16, y=1.1) # Adjust title position

    # Adjust layout 
    if plt is None:
        raise ImportError("Matplotlib is required but not installed.")
    plt.tight_layout()

    return axis