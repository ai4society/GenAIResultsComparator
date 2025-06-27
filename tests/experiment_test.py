from typing import Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gaico.experiment import DEFAULT_METRICS_TO_RUN, REGISTERED_METRICS, Experiment
from gaico.metrics.base import BaseMetric  # For mocking


#  Test Data and Fixtures
@pytest.fixture
def sample_llm_responses() -> Dict[str, str]:
    return {
        "ModelA": "This is a response from Model A.",
        "ModelB": "Model B provides this answer.",
        "ModelC": "Another response from Model C.",
    }


@pytest.fixture
def first_llm_response(sample_llm_responses) -> str:
    return list(sample_llm_responses.values())[0]


@pytest.fixture
def sample_reference_answer() -> str:
    return "This is the reference answer."


@pytest.fixture
def mock_metric_class_factory(monkeypatch):
    """Factory to create mock metric classes and instances."""
    created_mocks = {}

    def _factory(metric_name, score_to_return=0.5, sub_scores=None, init_raises=None):
        mock_metric_instance = MagicMock(spec=BaseMetric)
        if sub_scores:  # e.g. for ROUGE
            mock_metric_instance.calculate.return_value = sub_scores
        else:
            mock_metric_instance.calculate.return_value = score_to_return

        mock_metric_class = MagicMock(spec=type(BaseMetric))
        if init_raises:
            mock_metric_class.side_effect = init_raises  # To simulate ImportError on __init__
        else:
            mock_metric_class.return_value = mock_metric_instance

        original_metric = REGISTERED_METRICS.get(metric_name)
        monkeypatch.setitem(REGISTERED_METRICS, metric_name, mock_metric_class)
        # Store mock and original to ensure proper teardown if monkeypatch doesn't handle it fully
        # across multiple factory calls in one test (though usually it should).
        if metric_name not in created_mocks:
            created_mocks[metric_name] = (mock_metric_class, original_metric)

        return mock_metric_class, mock_metric_instance

    yield _factory

    # Teardown: Restore original REGISTERED_METRICS if modified by monkeypatch
    for metric_name, (mock_class, original_metric) in created_mocks.items():
        if original_metric is not None:
            monkeypatch.setitem(REGISTERED_METRICS, metric_name, original_metric)
        elif metric_name in REGISTERED_METRICS and REGISTERED_METRICS[metric_name] == mock_class:
            # If it was newly added by the mock and wasn't there originally
            monkeypatch.delitem(REGISTERED_METRICS, metric_name, raising=False)


#  Initialization Tests
def test_experiment_init_success(sample_llm_responses, sample_reference_answer):
    exp = Experiment(sample_llm_responses, sample_reference_answer)
    assert exp.llm_responses == sample_llm_responses
    assert exp.reference_answer == sample_reference_answer
    assert list(exp.models) == list(sample_llm_responses.keys())


def test_experiment_init_reference_answer_none_success(
    sample_llm_responses, first_llm_response, capsys
):
    exp = Experiment(sample_llm_responses, reference_answer=None)
    assert exp.llm_responses == sample_llm_responses
    assert exp.reference_answer == first_llm_response  # Should be ModelA's response
    assert list(exp.models) == list(sample_llm_responses.keys())

    captured = capsys.readouterr()
    first_model_name = list(sample_llm_responses.keys())[0]
    assert "Warning: reference_answer was not provided for Experiment." in captured.out
    assert f"Using the response from model '{first_model_name}' as the reference." in captured.out


def test_experiment_init_reference_answer_none_empty_llm_responses():
    with pytest.raises(
        ValueError, match="llm_responses cannot be empty if reference_answer is None"
    ):
        Experiment({}, reference_answer=None)


def test_experiment_init_invalid_llm_responses_type():
    with pytest.raises(TypeError, match="llm_responses must be a dictionary"):
        Experiment(["not", "a", "dict"], "ref")  # type: ignore


def test_experiment_init_invalid_llm_responses_content():
    with pytest.raises(ValueError, match="llm_responses must be Dict"):
        Experiment({1: "val"}, "ref")  # type: ignore
    with pytest.raises(ValueError, match="llm_responses must be Dict"):
        Experiment({"key": 123}, "ref")  # type: ignore


def test_experiment_init_invalid_reference_answer_type(sample_llm_responses):
    # Now allows None, so only test other invalid types
    with pytest.raises(TypeError, match="reference_answer must be a string or None"):
        Experiment(sample_llm_responses, ["not", "a", "string", "or_None"])  # type: ignore
    with pytest.raises(TypeError, match="reference_answer must be a string or None"):
        Experiment(sample_llm_responses, 123)  # type: ignore


#  to_dataframe() Tests
@pytest.mark.parametrize("ref_is_none", [False, True])
def test_to_dataframe_single_metric(
    sample_llm_responses,
    sample_reference_answer,
    first_llm_response,
    mock_metric_class_factory,
    ref_is_none,
):
    mock_jaccard_class, mock_jaccard_instance = mock_metric_class_factory(
        "Jaccard", score_to_return=0.7
    )

    current_ref = first_llm_response if ref_is_none else sample_reference_answer
    exp = Experiment(
        sample_llm_responses, reference_answer=None if ref_is_none else sample_reference_answer
    )

    df = exp.to_dataframe(metrics=["Jaccard"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_llm_responses)
    assert "Jaccard" in df["metric_name"].unique()
    assert mock_jaccard_instance.calculate.call_count == len(sample_llm_responses)

    # Check calculate was called with correct args for one model
    # The reference used should be `current_ref`
    mock_jaccard_instance.calculate.assert_any_call(sample_llm_responses["ModelA"], current_ref)
    mock_jaccard_instance.calculate.assert_any_call(sample_llm_responses["ModelB"], current_ref)
    assert all(df["score"] == 0.7)


@pytest.mark.parametrize("ref_is_none", [False, True])
def test_to_dataframe_multiple_metrics(
    sample_llm_responses,
    sample_reference_answer,
    first_llm_response,
    mock_metric_class_factory,
    ref_is_none,
):
    mock_j_class, mock_j_inst = mock_metric_class_factory("Jaccard", score_to_return=0.7)
    mock_r_class, mock_r_inst = mock_metric_class_factory(
        "ROUGE", sub_scores={"rouge1": 0.6, "rougeL": 0.5}
    )

    current_ref = first_llm_response if ref_is_none else sample_reference_answer
    exp = Experiment(
        sample_llm_responses, reference_answer=None if ref_is_none else sample_reference_answer
    )

    df = exp.to_dataframe(metrics=["Jaccard", "ROUGE"])

    assert len(df) == len(sample_llm_responses) * 3
    assert "Jaccard" in df["metric_name"].values
    assert "ROUGE_rouge1" in df["metric_name"].values
    assert "ROUGE_rougeL" in df["metric_name"].values

    assert mock_j_inst.calculate.call_count == len(sample_llm_responses)
    mock_j_inst.calculate.assert_any_call(sample_llm_responses["ModelA"], current_ref)

    assert mock_r_inst.calculate.call_count == len(sample_llm_responses)
    mock_r_inst.calculate.assert_any_call(sample_llm_responses["ModelA"], current_ref)


def test_to_dataframe_default_metrics(
    sample_llm_responses, sample_reference_answer, mock_metric_class_factory, monkeypatch
):
    # This test primarily checks if default metrics are picked up.
    # The reference handling is implicitly tested by the above parameterized tests.
    mock_j_class, _ = mock_metric_class_factory("Jaccard", 0.1)
    mock_l_class, _ = mock_metric_class_factory("Levenshtein", 0.2)

    original_defaults = list(DEFAULT_METRICS_TO_RUN)
    # Make DEFAULT_METRICS_TO_RUN predictable for this test
    monkeypatch.setattr("gaico.experiment.DEFAULT_METRICS_TO_RUN", ["Jaccard", "Levenshtein"])

    exp = Experiment(sample_llm_responses, sample_reference_answer)
    df = exp.to_dataframe()  # metrics=None

    assert len(df) == len(sample_llm_responses) * 2
    assert "Jaccard" in df["metric_name"].values
    assert "Levenshtein" in df["metric_name"].values

    # Restore DEFAULT_METRICS_TO_RUN
    monkeypatch.setattr("gaico.experiment.DEFAULT_METRICS_TO_RUN", original_defaults)


def test_to_dataframe_metric_init_fails(
    sample_llm_responses, sample_reference_answer, mock_metric_class_factory, capsys
):
    mock_j_class, mock_j_inst = mock_metric_class_factory("Jaccard", score_to_return=0.7)
    mock_b_class, _ = mock_metric_class_factory(
        "BERTScore", init_raises=ImportError("torch not found")
    )

    exp = Experiment(sample_llm_responses, sample_reference_answer)
    df = exp.to_dataframe(metrics=["Jaccard", "BERTScore"])

    captured = capsys.readouterr()
    assert (
        "Warning: Metric 'BERTScore' will be skipped due to missing dependencies: torch not found"
        in captured.out
    )
    assert "Jaccard" in df["metric_name"].values
    assert "BERTScore" not in df["metric_name"].values
    assert len(df) == len(sample_llm_responses)
    assert mock_j_inst.calculate.call_count == len(sample_llm_responses)


#  compare() Method Tests
@patch("gaico.experiment.viz_plt", None)  # Simulate matplotlib not installed for plotting part
@patch("gaico.experiment.generate_deltas_frame")  # Mock CSV generation
@patch("gaico.experiment.plot_metric_comparison")  # Mock bar plot
@patch("gaico.experiment.plot_radar_comparison")  # Mock radar plot
@pytest.mark.parametrize("ref_is_none", [False, True])
def test_compare_basic_run_no_plot_no_csv(
    mock_plot_radar,
    mock_plot_metric,
    mock_gen_deltas,
    sample_llm_responses,
    sample_reference_answer,
    first_llm_response,
    mock_metric_class_factory,
    ref_is_none,
):
    mock_j_class, mock_j_inst = mock_metric_class_factory("Jaccard", 0.7)

    current_ref = first_llm_response if ref_is_none else sample_reference_answer
    exp = Experiment(
        sample_llm_responses, reference_answer=None if ref_is_none else sample_reference_answer
    )

    results_df = exp.compare(metrics=["Jaccard"], plot=False, output_csv_path=None)

    assert isinstance(results_df, pd.DataFrame)
    assert "Jaccard" in results_df["metric_name"].values

    mock_j_inst.calculate.assert_any_call(sample_llm_responses["ModelA"], current_ref)

    mock_plot_metric.assert_not_called()
    mock_plot_radar.assert_not_called()
    mock_gen_deltas.assert_not_called()


@patch("gaico.experiment.viz_plt")  # Mock the alias used for plt
@patch("gaico.experiment.generate_deltas_frame")
@patch("gaico.experiment.plot_metric_comparison")
@patch("gaico.experiment.plot_radar_comparison")
def test_compare_with_single_metric_plot(
    mock_plot_radar,
    mock_plot_metric,
    mock_gen_deltas,
    mock_viz_plt,
    sample_llm_responses,
    sample_reference_answer,
    mock_metric_class_factory,
):
    mock_j_class, _ = mock_metric_class_factory("Jaccard", 0.7)
    mock_viz_plt.show = MagicMock()  # Mock plt.show()

    exp = Experiment(sample_llm_responses, sample_reference_answer)
    exp.compare(metrics=["Jaccard"], plot=True)

    mock_plot_metric.assert_called_once()
    call_args = mock_plot_metric.call_args[0]
    df_arg = call_args[0]
    assert isinstance(df_arg, pd.DataFrame)
    assert "Jaccard" in df_arg["metric_name"].values
    kwargs_arg = mock_plot_metric.call_args[1]
    assert kwargs_arg["metric_name"] == "Jaccard"

    mock_plot_radar.assert_not_called()
    mock_viz_plt.show.assert_called_once()  # Ensure plot is shown


@patch("gaico.experiment.viz_plt")
@patch("gaico.experiment.generate_deltas_frame")
@patch("gaico.experiment.plot_metric_comparison")
@patch("gaico.experiment.plot_radar_comparison")
def test_compare_with_multiple_metrics_radar_plot(
    mock_plot_radar,
    mock_plot_metric,
    mock_gen_deltas,
    mock_viz_plt,
    sample_llm_responses,
    sample_reference_answer,
    mock_metric_class_factory,
):
    mock_j_class, _ = mock_metric_class_factory("Jaccard", 0.7)
    mock_l_class, _ = mock_metric_class_factory("Levenshtein", 0.6)
    mock_s_class, _ = mock_metric_class_factory("SequenceMatcher", 0.5)  # Need 3+ for radar
    mock_viz_plt.show = MagicMock()

    exp = Experiment(sample_llm_responses, sample_reference_answer)
    exp.compare(metrics=["Jaccard", "Levenshtein", "SequenceMatcher"], plot=True)

    mock_plot_radar.assert_called_once()
    call_args_df = mock_plot_radar.call_args[0][0]
    assert isinstance(call_args_df, pd.DataFrame)
    assert "Jaccard" in call_args_df["metric_name"].values
    assert "Levenshtein" in call_args_df["metric_name"].values
    assert "SequenceMatcher" in call_args_df["metric_name"].values
    call_args_kwargs = mock_plot_radar.call_args[1]
    assert sorted(call_args_kwargs["metrics"]) == sorted(
        ["Jaccard", "Levenshtein", "SequenceMatcher"]
    )

    mock_plot_metric.assert_not_called()  # Should go to radar
    mock_viz_plt.show.assert_called_once()


@patch("gaico.experiment.viz_plt")
@patch("gaico.experiment.generate_deltas_frame")
@patch("gaico.experiment.plot_metric_comparison")
@patch("gaico.experiment.plot_radar_comparison")
def test_compare_with_two_metrics_bar_plots(  # Test the case where radar is skipped for <3 metrics
    mock_plot_radar,
    mock_plot_metric,
    mock_gen_deltas,
    mock_viz_plt,
    sample_llm_responses,
    sample_reference_answer,
    mock_metric_class_factory,
    capsys,
):
    mock_j_class, _ = mock_metric_class_factory("Jaccard", 0.7)
    mock_l_class, _ = mock_metric_class_factory("Levenshtein", 0.6)
    mock_viz_plt.show = MagicMock()

    exp = Experiment(sample_llm_responses, sample_reference_answer)
    exp.compare(metrics=["Jaccard", "Levenshtein"], plot=True)

    captured = capsys.readouterr()
    assert "Radar plot might not be informative. Generating bar plots instead." in captured.out
    assert mock_plot_metric.call_count == 2  # One bar plot per metric
    mock_plot_radar.assert_not_called()
    assert mock_viz_plt.show.call_count == 2


@patch("gaico.experiment.viz_plt", None)  # Matplotlib not available
@patch("gaico.experiment.generate_deltas_frame")
@patch("gaico.experiment.plot_metric_comparison")
@patch("gaico.experiment.plot_radar_comparison")
def test_compare_plot_true_but_matplotlib_unavailable(
    mock_plot_radar,
    mock_plot_metric,
    mock_gen_deltas,
    sample_llm_responses,
    sample_reference_answer,
    mock_metric_class_factory,
    capsys,
):
    mock_j_class, _ = mock_metric_class_factory("Jaccard", 0.7)
    exp = Experiment(sample_llm_responses, sample_reference_answer)
    exp.compare(metrics=["Jaccard"], plot=True)

    captured = capsys.readouterr()
    assert "Warning: Matplotlib/Seaborn are not installed. Skipping plotting." in captured.out
    mock_plot_metric.assert_not_called()


@patch("gaico.experiment.viz_plt")
@patch("gaico.experiment.generate_deltas_frame")
@patch("gaico.experiment.plot_metric_comparison")
@patch("gaico.experiment.plot_radar_comparison")
@pytest.mark.parametrize("ref_is_none", [False, True])
def test_compare_with_csv_output(
    mock_plot_radar,
    mock_plot_metric,
    mock_gen_deltas,
    mock_viz_plt,
    sample_llm_responses,
    sample_reference_answer,
    first_llm_response,
    mock_metric_class_factory,
    tmp_path,
    ref_is_none,
):
    mock_j_class, mock_j_inst = mock_metric_class_factory("Jaccard", 0.7)
    csv_path = tmp_path / "report.csv"

    current_ref_text_for_csv = first_llm_response if ref_is_none else sample_reference_answer
    exp = Experiment(
        sample_llm_responses, reference_answer=None if ref_is_none else sample_reference_answer
    )

    exp.compare(metrics=["Jaccard"], output_csv_path=str(csv_path))

    mock_gen_deltas.assert_called_once()
    call_kwargs = mock_gen_deltas.call_args[1]
    assert call_kwargs["output_csv_path"] == str(csv_path)

    # Check that the reference texts passed to generate_deltas_frame are correct
    # It should be a list of the same reference repeated for each model
    expected_ref_texts_for_csv = [current_ref_text_for_csv] * len(sample_llm_responses)
    assert call_kwargs["reference_texts"] == expected_ref_texts_for_csv

    # Check structure of threshold_results
    threshold_results_arg = call_kwargs["threshold_results"]
    assert isinstance(threshold_results_arg, list)
    assert len(threshold_results_arg) == len(sample_llm_responses)
    first_model_results = threshold_results_arg[0]
    assert "Jaccard" in first_model_results
    assert "score" in first_model_results["Jaccard"]


def test_compare_no_runnable_metrics(
    sample_llm_responses, sample_reference_answer, mock_metric_class_factory, capsys
):
    mock_metric_class_factory("Jaccard", init_raises=ImportError("fail J"))
    mock_metric_class_factory("ROUGE", init_raises=ImportError("fail R"))

    exp = Experiment(sample_llm_responses, sample_reference_answer)
    result_df = exp.compare(metrics=["Jaccard", "ROUGE"])

    captured = capsys.readouterr()
    assert "None of the specified metrics are runnable" in captured.out
    assert result_df is None


def test_compare_custom_thresholds(
    sample_llm_responses, sample_reference_answer, mock_metric_class_factory, tmp_path
):
    mock_j_class, _ = mock_metric_class_factory("Jaccard", 0.7)
    mock_r_class, _ = mock_metric_class_factory("ROUGE", sub_scores={"rouge1": 0.4})
    csv_path = tmp_path / "report_custom_thresh.csv"

    with patch("gaico.experiment.generate_deltas_frame") as mock_gen_deltas:
        exp = Experiment(sample_llm_responses, sample_reference_answer)
        exp.compare(
            metrics=["Jaccard", "ROUGE"],
            custom_thresholds={"Jaccard": 0.8, "ROUGE_rouge1": 0.3},
            output_csv_path=str(csv_path),
        )
        mock_gen_deltas.assert_called_once()
        threshold_results_arg = mock_gen_deltas.call_args[1]["threshold_results"]

        # Check ModelA's results (assuming order or find by model if needed)
        model_a_results = threshold_results_arg[0]  # Assuming ModelA is first
        assert model_a_results["Jaccard"]["score"] == 0.7
        assert model_a_results["Jaccard"]["threshold_applied"] == 0.8
        assert model_a_results["Jaccard"]["passed_threshold"] is False

        assert model_a_results["ROUGE_rouge1"]["score"] == 0.4
        assert model_a_results["ROUGE_rouge1"]["threshold_applied"] == 0.3
        assert model_a_results["ROUGE_rouge1"]["passed_threshold"] is True
