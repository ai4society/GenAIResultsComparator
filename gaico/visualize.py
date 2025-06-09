import warnings
from typing import Any, Callable, Optional

import pandas

# Optional Imports - Keep these for lazy loading
pi: Optional[float]
plt: Optional[Any]
np: Optional[Any]
pd: Optional[Any]
sns: Optional[Any]

# Initialize all to None.
pi = None
plt = None
np = None
pd = None
sns = None

try:
    from math import pi as _math_pi_temp  # Import the value of pi using a temporary alias

    import matplotlib.pyplot as _plt_temp  # Import pyplot using a temporary alias

    pi = _math_pi_temp  # Assign the imported value to the module-level 'pi'
    plt = _plt_temp  # Assign the imported module to the module-level 'plt'
except ImportError:
    # If either import in this block fails, both are set to None,
    # maintaining the original logic.
    plt = None
    pi = 3.14159

try:
    import numpy as _np_temp  # Import numpy using a temporary alias

    np = _np_temp  # Assign to the module-level 'np'
except ImportError:
    # Allow module import, functions will raise runtime error if called without numpy
    np = None

try:
    import pandas as _pd_temp  # Import pandas using a temporary alias

    pd = _pd_temp  # Assign to the module-level 'pd'
except ImportError:
    # Allow module import, functions will raise runtime error if called without pandas
    pd = None

try:
    import seaborn as _sns_temp  # Import seaborn using a temporary alias

    sns = _sns_temp  # Assign to the module-level 'sns'
except ImportError:
    # Allow module import, functions will raise runtime error if called without seaborn
    sns = None


# Bar Plot Function for comparing single metric across models
def plot_metric_comparison(
    df: pandas.DataFrame,
    aggregate_func: Optional[Callable] = None,
    **kwargs: Any,
) -> Any:
    """
    Generates a bar plot comparing different models based on a single metric,
    after aggregating scores using the provided aggregate_func.

    :param df: DataFrame containing the scores, typically from prepare_results_dataframe.
        Expected columns are defined by model_col, metric_col, and score_col in kwargs.
    :type df: pd.DataFrame
    :param aggregate_func: A function to aggregate scores (e.g., numpy.mean, numpy.median).
        Defaults to numpy.mean if None.
    :type aggregate_func: Optional[Callable]
    :param kwargs: Additional keyword arguments:
        - metric_name (str, required): The name of the metric to plot.
        - model_col (str, optional): Name of the column identifying models. Defaults to "model_name".
        - score_col (str, optional): Name of the column containing scores. Defaults to "score".
        - metric_col (str, optional): Name of the column containing metric names. Defaults to "metric_name".
        - title (Optional[str], optional): Title for the plot.
        - xlabel (Optional[str], optional): Label for the x-axis. Defaults to "Model".
        - ylabel (Optional[str], optional): Label for the y-axis. Defaults to the plotted metric's name.
        - figsize (tuple, optional): Figure size. Defaults to (10, 6).
        - axis (Optional[matplotlib.axes.Axes], optional): Matplotlib Axes to plot on.
        - Other kwargs are passed to seaborn.barplot.
    :type kwargs: Any
    :raises ImportError: If required libraries (matplotlib, seaborn, pandas, numpy) are not installed.
    :raises ValueError: If 'metric_name' is not provided in kwargs.
    :return: The matplotlib Axes object containing the plot.
    :rtype: matplotlib.axes.Axes
    """

    if pd is None:
        raise ImportError("Pandas is required but not installed.")
    if np is None:
        raise ImportError("Numpy is required but not installed.")
    if sns is None:
        raise ImportError("Seaborn is required but not installed.")
    if plt is None:
        raise ImportError("Matplotlib is required but not installed.")

    actual_aggregate_func = aggregate_func
    if actual_aggregate_func is None:
        if np is None:
            raise ImportError(
                "Numpy is required for default aggregation. Please install numpy or provide an aggregate_func."
            )
        actual_aggregate_func = np.mean

    metric_to_plot = kwargs.pop("metric_name", None)
    if metric_to_plot is None:
        raise ValueError(
            "'metric_name' must be provided in kwargs to specify which metric to plot."
        )

    model_col = kwargs.pop("model_col", "model_name")
    score_col = kwargs.pop("score_col", "score")
    metric_col = kwargs.pop("metric_col", "metric_name")

    title_val = kwargs.pop("title", None)
    xlabel_val = kwargs.pop("xlabel", "Model")
    ylabel_val = kwargs.pop("ylabel", None)
    figsize_val = kwargs.pop("figsize", (10, 6))
    axis_val = kwargs.pop("axis", None)

    # Filter for the specific metric
    metric_specific_df = df[df[metric_col] == metric_to_plot]

    # Aggregate scores
    aggregated_data = metric_specific_df.groupby(model_col, as_index=False)[score_col].apply(
        actual_aggregate_func
    )
    # If aggregate_func returns a DataFrame (i.e from a custom function), make sure it has the correct columns
    if (
        not isinstance(aggregated_data, pd.DataFrame)
        or model_col not in aggregated_data.columns
        or score_col not in aggregated_data.columns
    ):
        # Fallback for simple aggregation like np.mean.
        if actual_aggregate_func == np.mean:
            aggregated_data = (
                metric_specific_df.groupby(model_col)[score_col].agg("mean").reset_index()
            )
        else:
            aggregated_data = (
                metric_specific_df.groupby(model_col)[score_col]
                .agg(actual_aggregate_func)
                .reset_index()
            )

    current_axis = axis_val
    if current_axis is None:
        fig, current_axis = plt.subplots(figsize=figsize_val)

    sns.barplot(data=aggregated_data, x=model_col, y=score_col, ax=current_axis, **kwargs)

    plot_title = title_val if title_val else f"{metric_to_plot} Comparison"
    plot_ylabel_text = ylabel_val if ylabel_val else metric_to_plot

    current_axis.set_title(plot_title)
    current_axis.set_xlabel(xlabel_val)
    current_axis.set_ylabel(plot_ylabel_text)

    if len(aggregated_data[model_col].unique()) > 5:
        current_axis.tick_params(axis="x", rotation=45)
    else:
        current_axis.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    return current_axis


# Radar Plot Function for comparing multiple metrics across models
def plot_radar_comparison(
    df: pandas.DataFrame,
    aggregate_func: Optional[Callable] = None,
    **kwargs: Any,
) -> Any:
    """
    Generates a radar plot comparing multiple models across several metrics,
    after aggregating scores using the provided aggregate_func.

    :param df: DataFrame containing the scores in long format, typically from prepare_results_dataframe.
        Expected columns are defined by model_col, metric_col, and score_col in kwargs.
    :type df: pd.DataFrame
    :param aggregate_func: A function to aggregate scores (e.g., numpy.mean, numpy.median).
        Defaults to numpy.mean if None.
    :type aggregate_func: Optional[Callable]
    :param kwargs: Additional keyword arguments:
        - metrics (List[str], optional): List of metric names to include. If None, all metrics in df are used.
        - model_col (str, optional): Name of the column identifying models. Defaults to "model_name".
        - score_col (str, optional): Name of the column containing scores. Defaults to "score".
        - metric_col (str, optional): Name of the column containing metric names. Defaults to "metric_name".
        - title (Optional[str], optional): Title for the plot. Defaults to "Model Comparison Radar Plot".
        - figsize (tuple, optional): Figure size. Defaults to (8, 8).
        - fill_alpha (float, optional): Alpha for filled area. Defaults to 0.1.
        - line_width (float, optional): Width of plot lines. Defaults to 1.0.
        - y_ticks (Optional[List[float]], optional): Custom y-axis ticks.
        - axis (Optional[matplotlib.axes.Axes], optional): Matplotlib polar Axes to plot on.
    :type kwargs: Any
    :raises ImportError: If required libraries (matplotlib, numpy, pandas) are not installed.
    :raises ValueError: If aggregation results in no data or metrics.
    :return: The matplotlib Axes object containing the plot.
    :rtype: matplotlib.axes.Axes
    """
    if pd is None:
        raise ImportError("Pandas is required but not installed.")
    if np is None:
        raise ImportError("Numpy is required but not installed.")
    if plt is None:
        raise ImportError("Matplotlib is required but not installed.")
    if pi is None:
        raise ImportError("math.pi is required from math module but not available.")

    actual_aggregate_func = aggregate_func
    if actual_aggregate_func is None:
        actual_aggregate_func = np.mean

    model_col = kwargs.pop("model_col", "model_name")
    metric_col = kwargs.pop("metric_col", "metric_name")
    score_col = kwargs.pop("score_col", "score")

    title_val = kwargs.pop("title", "Model Comparison Radar Plot")
    figsize_val = kwargs.pop("figsize", (8, 8))
    fill_alpha_val = kwargs.pop("fill_alpha", 0.1)
    line_width_val = kwargs.pop("line_width", 1.0)
    y_ticks_val = kwargs.pop("y_ticks", None)
    axis_val = kwargs.pop("axis", None)
    metrics_list_user = kwargs.pop("metrics", None)

    # Aggregate data
    if actual_aggregate_func == np.mean:
        aggregated_scores = df.groupby([model_col, metric_col])[score_col].agg("mean")
    else:
        aggregated_scores = df.groupby([model_col, metric_col])[score_col].agg(
            actual_aggregate_func
        )
    pivot_df = aggregated_scores.unstack(level=metric_col)

    if pivot_df.empty:
        warnings.warn(
            "Pivot table is empty after aggregation. Cannot generate radar plot.",
            UserWarning,
        )
        return None
    if metrics_list_user:
        # Keep only metrics present in both user list and pivot_df columns
        metrics_to_plot = [m for m in metrics_list_user if m in pivot_df.columns]
        if not metrics_to_plot:
            raise ValueError(
                "None of the specified 'metrics' are available in the data after aggregation."
            )
        pivot_df = pivot_df[metrics_to_plot]
    else:
        metrics_to_plot = pivot_df.columns.tolist()

    if not metrics_to_plot:
        raise ValueError("No metrics available to plot after aggregation and filtering.")

    models = pivot_df.index.tolist()
    num_vars = len(metrics_to_plot)

    if num_vars < 1:  # Should be caught by earlier checks, but as a safeguard
        raise ValueError("At least one metric is required for a radar plot.")

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    current_axis = axis_val
    if current_axis is None:
        fig, current_axis = plt.subplots(figsize=figsize_val, subplot_kw=dict(polar=True))
    elif not hasattr(current_axis, "set_theta_offset"):  # Check if it's a polar axis
        raise ValueError("Provided 'axis' must be a polar projection for radar plot.")

    current_axis.set_xticks(angles[:-1])
    current_axis.set_xticklabels(metrics_to_plot)

    if y_ticks_val:
        current_axis.set_yticks(y_ticks_val)
    else:
        max_val = pivot_df.max().max()
        if pd.notna(max_val) and max_val > 0:
            upper_lim = np.ceil(max_val * 1.1 * 10) / 10  # Adds some padding
            step = max(0.1, np.round(upper_lim / 5, 1))
            step = step if step > 0 else 0.1
            current_axis.set_yticks(np.arange(0, upper_lim + step, step=step))
        elif pd.notna(max_val) and max_val == 0:  # Handle case where all values are 0
            current_axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:  # Default if max_val is NaN (e.g. all NaNs in pivot_df) or negative (less common for scores)
            current_axis.set_yticks(np.arange(0, 1.1, 0.2))

    angles_np = np.array(angles)

    for model in models:
        values = pivot_df.loc[model].values.flatten().tolist()
        values_closed = values + values[:1]
        values_masked = np.array(values_closed, dtype=float)

        current_axis.plot(
            angles_np,
            values_masked,
            linewidth=line_width_val,
            linestyle="solid",
            label=model,
            **kwargs,
        )
        current_axis.fill(angles_np, values_masked, alpha=fill_alpha_val, **kwargs)

    current_axis.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    if title_val:
        current_axis.set_title(title_val, size=16, y=1.1)

    plt.tight_layout()
    return current_axis
