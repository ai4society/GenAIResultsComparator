from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import the module to be tested
from gaico import visualize


# Sample DataFrame for plotting tests
@pytest.fixture
def sample_plot_df():
    return pd.DataFrame(
        {
            "model_name": ["ModelA", "ModelA", "ModelB", "ModelB", "ModelA", "ModelB"],
            "metric_name": ["BLEU", "ROUGE_rouge1", "BLEU", "ROUGE_rouge1", "Jaccard", "Jaccard"],
            "score": [0.8, 0.75, 0.7, 0.65, 0.9, 0.85],
        }
    )


# ** Tests for plot_metric_comparison (Bar Plot) **
@patch("gaico.visualize.sns")  # Mock seaborn
@patch("gaico.visualize.plt")  # Mock matplotlib.pyplot
def test_plot_metric_comparison_runs(mock_plt, mock_sns, sample_plot_df):
    """Test that plot_metric_comparison calls seaborn.barplot and plt.show/tight_layout."""
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)  # fig, ax

    visualize.plot_metric_comparison(sample_plot_df, metric_name="BLEU")

    mock_sns.barplot.assert_called_once()
    # Check some key arguments passed to barplot
    call_args = mock_sns.barplot.call_args[1]  # kwargs
    assert call_args["x"] == "model_name"
    assert call_args["y"] == "score"
    assert call_args["ax"] == mock_ax
    assert isinstance(call_args["data"], pd.DataFrame)

    mock_ax.set_title.assert_called_once_with("BLEU Comparison")
    mock_ax.set_xlabel.assert_called_once_with("Model")
    mock_ax.set_ylabel.assert_called_once_with("BLEU")
    mock_plt.tight_layout.assert_called_once()
    # mock_plt.show() # If you want to assert show is called, uncomment


@patch("gaico.visualize.sns")
@patch("gaico.visualize.plt")
def test_plot_metric_comparison_custom_params(mock_plt, mock_sns, sample_plot_df):
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    visualize.plot_metric_comparison(
        sample_plot_df,
        metric_name="Jaccard",
        title="Custom Jaccard Plot",
        xlabel="Models",
        ylabel="Jaccard Index",
        model_col="model_name",  # Default, but good to be explicit
        score_col="score",
        aggregate_func=np.median,
    )
    mock_sns.barplot.assert_called_once()
    call_args_data = mock_sns.barplot.call_args[1]["data"]
    # Verify aggregation happened (difficult to check exact values without running np.median here)
    # but we can check that the data passed to barplot is aggregated
    assert (
        len(call_args_data) == sample_plot_df["model_name"].nunique()
    )  # Should be one row per model

    mock_ax.set_title.assert_called_with("Custom Jaccard Plot")
    mock_ax.set_xlabel.assert_called_with("Models")
    mock_ax.set_ylabel.assert_called_with("Jaccard Index")


@patch("gaico.visualize.plt")
def test_plot_metric_comparison_missing_metric_name(mock_plt, sample_plot_df):
    with pytest.raises(ValueError, match="'metric_name' must be provided"):
        visualize.plot_metric_comparison(sample_plot_df)  # Missing metric_name


# Test for ImportError if dependencies are missing (requires more setup to simulate)
# You can mock the import attempts at the top of visualize.py
@patch("gaico.visualize.pd", None)
def test_plot_metric_comparison_missing_pandas(sample_plot_df):
    with pytest.raises(ImportError, match="Pandas is required"):
        visualize.plot_metric_comparison(sample_plot_df, metric_name="BLEU")


# ** Tests for plot_radar_comparison **
@patch("gaico.visualize.plt")  # Mock matplotlib.pyplot
@patch("gaico.visualize.np", np)  # Use actual numpy for calculations if needed by radar logic
@patch("gaico.visualize.pd", pd)  # Use actual pandas
def test_plot_radar_comparison_runs(mock_plt, sample_plot_df):
    """Test that plot_radar_comparison calls plt.subplots and relevant Axes methods."""
    mock_ax_polar = MagicMock()
    mock_ax_polar.set_theta_offset = MagicMock()  # Simulate polar axis
    mock_plt.subplots.return_value = (MagicMock(), mock_ax_polar)

    metrics_to_plot = ["BLEU", "ROUGE_rouge1"]
    visualize.plot_radar_comparison(sample_plot_df, metrics=metrics_to_plot)

    mock_plt.subplots.assert_called_once_with(figsize=(8, 8), subplot_kw=dict(polar=True))
    mock_ax_polar.set_xticks.assert_called_once()
    mock_ax_polar.set_xticklabels.assert_called_once_with(metrics_to_plot)
    mock_ax_polar.plot.assert_called()  # Should be called for each model
    assert mock_ax_polar.plot.call_count == sample_plot_df["model_name"].nunique()
    mock_ax_polar.fill.assert_called()
    mock_ax_polar.legend.assert_called_once()
    mock_plt.tight_layout.assert_called_once()


@patch("gaico.visualize.plt")
def test_plot_radar_comparison_no_metrics_after_agg(mock_plt, sample_plot_df):
    # Create data where specified metrics don't exist or result in empty pivot
    empty_df = pd.DataFrame(columns=["model_name", "metric_name", "score"])
    with pytest.warns(
        UserWarning, match="Pivot table is empty"
    ):  # or ValueError depending on exact flow
        visualize.plot_radar_comparison(empty_df, metrics=["NonExistent"])

    df_no_matching_metrics = sample_plot_df[sample_plot_df["metric_name"] == "BLEU"]
    with pytest.raises(ValueError, match="None of the specified 'metrics' are available"):
        visualize.plot_radar_comparison(df_no_matching_metrics, metrics=["ROUGE_rouge1"])


@patch("gaico.visualize.plt", None)  # Simulate matplotlib not being installed
def test_plot_radar_missing_matplotlib(sample_plot_df):
    with pytest.raises(ImportError, match="Matplotlib is required"):
        visualize.plot_radar_comparison(sample_plot_df, metrics=["BLEU"])
