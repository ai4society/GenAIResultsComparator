from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from gaico.metrics import (
    BLEU,
    ROUGE,
    BERTScore,
    CosineSimilarity,
    JaccardSimilarity,
    JSDivergence,
    LevenshteinDistance,
    SequenceMatcherSimilarity,
)

from .metrics.base import BaseMetric
from .thresholds import apply_thresholds, get_default_thresholds
from .utils import generate_deltas_frame, prepare_results_dataframe

# Import plt from visualize to check availability and use for showing plots
# Use an alias to avoid conflict if user imports plt
from .visualize import plot_metric_comparison, plot_radar_comparison
from .visualize import plt as viz_plt

REGISTERED_METRICS: Dict[str, type[BaseMetric]] = {
    "Jaccard": JaccardSimilarity,
    "Cosine": CosineSimilarity,
    "Levenshtein": LevenshteinDistance,
    "SequenceMatcher": SequenceMatcherSimilarity,
    "BLEU": BLEU,
    "ROUGE": ROUGE,
    "JSD": JSDivergence,
    "BERTScore": BERTScore,
}

DEFAULT_METRICS_TO_RUN = list(REGISTERED_METRICS.keys())


class Experiment:
    """
    An abstraction to simplify plotting, applying thresholds, and generating CSVs
    for comparing LLM responses against reference answers using various metrics.
    """

    def __init__(
        self,
        llm_responses: Dict[str, str],
        reference_answer: str,
    ):
        """
        Initializes the Experiment.

        :param llm_responses: A dictionary mapping model names (str) to their generated text responses (str).
        :type llm_responses: Dict[str, str]
        :param reference_answer: A single reference text (str) to compare against.
        :type reference_answer: str
        :raises TypeError: If llm_responses is not a dictionary or reference_answer is not a string.
        :raises ValueError: If llm_responses does not contain string keys and values.
        """
        if not isinstance(llm_responses, dict):
            raise TypeError("llm_responses must be a dictionary.")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in llm_responses.items()):
            raise ValueError("llm_responses must be Dict[str, str] (model_name -> text).")
        if not isinstance(reference_answer, str):
            raise TypeError("reference_answer must be a string.")

        self.llm_responses = llm_responses
        self.reference_answer = reference_answer
        self.models = list(llm_responses.keys())

        # model_name -> base_metric_name -> score_value
        self._raw_scores: Dict[str, Dict[str, Any]] = {}
        # Caches the full, unfiltered DF from _raw_scores
        self._results_df_cache: Optional[pd.DataFrame] = None
        # model_name -> {flat_metric: {details}}
        self._thresholded_results_cache: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None

    def _calculate_scores_for_metric(self, metric_name: str) -> None:
        """
        Calculates scores for a given base metric across all models if not already done.

        :param metric_name: The name of the metric to calculate (e.g., "Jaccard", "ROUGE").
        :type metric_name: str
        :return: None
        :rtype: None
        """
        if metric_name not in REGISTERED_METRICS:
            # This should be caught by callers, but as a safeguard:
            raise ValueError(f"Metric '{metric_name}' is not registered.")

        metric_cls = REGISTERED_METRICS[metric_name]
        # Use default initialization for metric classes
        try:
            metric_instance = metric_cls()
        except ImportError as e:
            # This ImportError is raised by metric's __init__ if deps are missing
            print(
                f"Warning: Metric '{metric_name}' cannot be initialized due to missing dependencies and will be skipped. Details: {e}"
            )
            self._raw_scores.setdefault(self.models[0], {})[metric_name] = (
                None  # Mark as attempted but failed
            )
            return  # Skip calculation for this metric

        for model_name, gen_text in self.llm_responses.items():
            if model_name not in self._raw_scores:
                self._raw_scores[model_name] = {}

            if metric_name not in self._raw_scores[model_name]:  # Calculate if not present
                score = metric_instance.calculate(gen_text, self.reference_answer)
                self._raw_scores[model_name][metric_name] = (
                    score  # Score can be None if metric_instance failed init earlier
                )
                self._results_df_cache = None  # Invalidate DataFrame cache
                self._thresholded_results_cache = None  # Invalidate threshold cache

    def _get_runnable_metrics(self, requested_metrics: List[str]) -> List[str]:
        """
        Filters a list of requested metric names, returning only those
        that can be successfully instantiated (i.e., their dependencies are met).
        """
        runnable = []
        for metric_name in requested_metrics:
            if metric_name not in REGISTERED_METRICS:
                print(f"Warning: Metric '{metric_name}' is not registered and will be skipped.")
                continue

            metric_cls = REGISTERED_METRICS[metric_name]
            try:
                # Attempt to instantiate to check for ImportErrors from __init__
                _ = metric_cls()
                runnable.append(metric_name)
            except ImportError as e:
                print(
                    f"Warning: Metric '{metric_name}' will be skipped due to missing dependencies: {e}"
                )
            except Exception as e:  # Catch other potential init errors
                print(
                    f"Warning: Metric '{metric_name}' failed to initialize and will be skipped: {e}"
                )
        return runnable

    def _ensure_scores_calculated(self, base_metrics_to_calculate: List[str]):
        for metric_name in base_metrics_to_calculate:
            self._calculate_scores_for_metric(metric_name)

    def _get_internal_scores_df(self) -> pd.DataFrame:
        """
        Ensures all raw scores are converted to a DataFrame and caches it.
        This DataFrame contains all calculated metrics, flattened.
        """
        if self._results_df_cache is None:
            if not self._raw_scores:  # No scores calculated yet
                # This might happen if _get_scores_df is called before any metric calculation.
                # Calculate all default metrics as a baseline.
                # Ensure only runnable default metrics are calculated
                runnable_default_metrics = self._get_runnable_metrics(DEFAULT_METRICS_TO_RUN)
                self._ensure_scores_calculated(runnable_default_metrics)

            self._results_df_cache = prepare_results_dataframe(self._raw_scores)
        return (
            self._results_df_cache.copy() if self._results_df_cache is not None else pd.DataFrame()
        )

    def _get_filtered_scores_df(self, base_metrics_to_include: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame of scores, filtered to include only metrics
        derived from the provided list of base_metrics_to_include.
        Calculates scores if necessary.

        :param base_metrics_to_include: A list of base metric names (e.g., "Jaccard", "ROUGE").
        :type base_metrics_to_include: List[str]
        :return: A pandas DataFrame with columns "model_name", "metric_name", "score".
                 "metric_name" will contain flat metric names (e.g., "ROUGE_rouge1").
        :rtype: pd.DataFrame
        """
        # Ensure scores are calculated only for runnable metrics from the include list
        self._ensure_scores_calculated(base_metrics_to_include)

        full_df = self._get_internal_scores_df()  # Gets the cached or newly prepared full DF

        if full_df.empty:
            return pd.DataFrame(columns=["model_name", "metric_name", "score"])

        # Filter the full_df to include only flat metrics corresponding to base_metrics_to_include
        flat_metrics_to_keep = []
        all_df_metric_names = full_df["metric_name"].unique()

        # Iterate over the originally requested & runnable metrics
        for base_name in base_metrics_to_include:
            for df_m_name in all_df_metric_names:  # df_m_name can be 'ROUGE' or 'ROUGE_rouge1'
                if df_m_name == base_name or df_m_name.startswith(base_name + "_"):
                    flat_metrics_to_keep.append(df_m_name)

        if not flat_metrics_to_keep:  # No metrics matched
            return pd.DataFrame(columns=["model_name", "metric_name", "score"])

        return full_df[full_df["metric_name"].isin(list(set(flat_metrics_to_keep)))].copy()

    def to_dataframe(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame of scores for the specified metrics.
        If metrics is None, scores for all default metrics are returned.

        :param metrics: A list of base metric names (e.g., "Jaccard", "ROUGE"). Defaults to None.
        :type metrics: Optional[List[str]]
        :return: A pandas DataFrame with columns "model_name", "metric_name", "score".
                 "metric_name" will contain flat metric names (e.g., "ROUGE_rouge1").
        :rtype: pd.DataFrame
        """
        requested_metrics = metrics if metrics is not None else DEFAULT_METRICS_TO_RUN

        # Filter to only metrics that can actually run (dependencies met)
        runnable_metrics = self._get_runnable_metrics(requested_metrics)

        return self._get_filtered_scores_df(base_metrics_to_include=runnable_metrics)

    def _get_thresholded_results(
        self,
        flat_metrics_for_thresholding: List[str],
        custom_thresholds: Optional[Dict[str, float]],
        scores_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Calculates and returns thresholded results for all models.
        Assumes scores_df contains the necessary flat_metrics.
        :param flat_metrics_for_thresholding: List of flat metric names to apply thresholds to.
        :type flat_metrics_for_thresholding: List[str]
        :param custom_thresholds: Optional dictionary mapping flat metric names to custom threshold values.
                                  If provided, these will override default thresholds.
        :type custom_thresholds: Optional[Dict[str, float]]
        :param scores_df: DataFrame containing scores with columns "model_name", "metric_name", "score".
                          This should already be filtered to include only relevant flat metrics.
        :type scores_df: pd.DataFrame
        :return: A dictionary mapping model names to their thresholded results.
                 Each model's results are a dictionary of flat metric names to their score and pass/fail status.
        :rtype: Dict[str, Dict[str, Dict[str, Any]]]
        """
        # Determine effective thresholds for each flat_metric_name
        default_threshold_map = get_default_thresholds()

        # This will store model_name -> {flat_metric_name: {score: val, passed: bool, ...}}
        all_models_thresholded_output: Dict = {}

        for model_name in self.models:
            model_scores_df = scores_df[
                (scores_df["model_name"] == model_name)
                & (scores_df["metric_name"].isin(flat_metrics_for_thresholding))
            ]

            if model_scores_df.empty:
                all_models_thresholded_output[model_name] = {}
                continue

            scores_for_model_dict = pd.Series(
                model_scores_df.score.values, index=model_scores_df.metric_name
            ).to_dict()

            # Determine thresholds to apply for this model's scores
            thresholds_for_this_model_apply = {}
            for flat_name, score_val in scores_for_model_dict.items():
                base_name_candidate = flat_name.split("_")[0]
                chosen_threshold_value = None

                if custom_thresholds:
                    if flat_name in custom_thresholds:
                        chosen_threshold_value = custom_thresholds[flat_name]
                    elif (
                        base_name_candidate in custom_thresholds and chosen_threshold_value is None
                    ):
                        chosen_threshold_value = custom_thresholds[base_name_candidate]

                if chosen_threshold_value is None:  # Fallback to defaults
                    # Defaults usually use base names.
                    if base_name_candidate in default_threshold_map:
                        chosen_threshold_value = default_threshold_map[base_name_candidate]
                    elif flat_name in default_threshold_map:  # Less common for defaults
                        chosen_threshold_value = default_threshold_map[flat_name]

                if chosen_threshold_value is not None:
                    thresholds_for_this_model_apply[flat_name] = chosen_threshold_value

            thresholded_for_model = apply_thresholds(
                scores_for_model_dict, thresholds_for_this_model_apply
            )
            all_models_thresholded_output[model_name] = thresholded_for_model

        self._thresholded_results_cache = all_models_thresholded_output  # Cache this
        return all_models_thresholded_output

    def compare(
        self,
        metrics: Optional[List[str]] = None,
        plot: bool = False,
        custom_thresholds: Optional[Dict[str, float]] = None,
        output_csv_path: Optional[str] = None,
        aggregate_func: Optional[Callable] = None,
        plot_title_suffix: str = "Comparison",
        radar_metrics_limit: int = 12,
    ) -> Optional[pd.DataFrame]:
        """
        Compares models based on specified metrics, optionally plotting and generating a CSV.

        :param metrics: List of base metric names. If None, uses all default registered metrics.
        :type metrics: Optional[List[str]]
        :param plot: If True, generates and shows plots. Defaults to False.
        :type plot: bool
        :param custom_thresholds: Dictionary of metric names (base or flat) to threshold values.
                                  Overrides default thresholds.
        :type custom_thresholds: Optional[Dict[str, float]]
        :param output_csv_path: If provided, path to save a CSV report of thresholded results.
        :type output_csv_path: Optional[str]
        :param aggregate_func: Aggregation function (e.g., np.mean, np.median) for plotting
                                when multiple scores exist per model/metric (not typical for default setup).
        :type aggregate_func: Optional[Callable]
        :param plot_title_suffix: Suffix for plot titles.
        :type plot_title_suffix: str
        :param radar_metrics_limit: Maximum number of metrics for a radar plot to maintain readability.
        :type radar_metrics_limit: int
        :return: A pandas DataFrame containing the scores for the compared metrics, or None if no valid metrics.
        :rtype: Optional[pd.DataFrame]
        """
        requested_base_metrics = metrics if metrics is not None else DEFAULT_METRICS_TO_RUN

        # Determine which of the requested metrics are actually runnable based on available dependencies
        runnable_base_metrics = self._get_runnable_metrics(requested_base_metrics)

        if not runnable_base_metrics:
            if metrics:  # User specified metrics, but none were runnable
                print(
                    "Warning: None of the specified metrics are runnable due to missing dependencies or registration. Aborting compare."
                )
            else:  # Default metrics were requested, but none are runnable
                print(
                    "Warning: No default metrics are runnable due to missing dependencies. Aborting compare."
                )
            return None

        # 1. Get DataFrame of scores (flat metric names)
        # This will calculate scores for the runnable_base_metrics.
        current_scores_df = self._get_filtered_scores_df(
            base_metrics_to_include=runnable_base_metrics
        )

        if current_scores_df.empty:
            print("No results to compare after processing metrics.")
            return current_scores_df  # Return empty DF

        # These are the actual (potentially flattened) metric names present in the DataFrame
        actual_flat_metrics_in_df = sorted(list(current_scores_df["metric_name"].unique()))

        # 2. Apply Thresholds
        thresholded_results_all_models = self._get_thresholded_results(
            flat_metrics_for_thresholding=actual_flat_metrics_in_df,
            custom_thresholds=custom_thresholds,
            scores_df=current_scores_df,
        )

        # 3. Plotting
        if plot:
            if viz_plt is None:
                print("Warning: Matplotlib/Seaborn are not installed. Skipping plotting.")
            else:
                num_actual_metrics = len(actual_flat_metrics_in_df)
                # num_models = len(self.models) # Not directly used for plot choice here

                if num_actual_metrics == 0:
                    print("No metrics available for plotting.")
                elif num_actual_metrics == 1:
                    metric_to_plot = actual_flat_metrics_in_df[0]
                    plot_metric_comparison(
                        current_scores_df,
                        metric_name=metric_to_plot,
                        aggregate_func=aggregate_func,
                        title=f"{metric_to_plot} {plot_title_suffix}",
                    )
                    viz_plt.show()
                else:  # Multiple metrics
                    metrics_for_radar = actual_flat_metrics_in_df
                    if len(actual_flat_metrics_in_df) > radar_metrics_limit:
                        print(
                            f"Warning: Too many metrics ({len(actual_flat_metrics_in_df)}) for radar plot. Plotting first {radar_metrics_limit}."
                            "To change this, adjust radar_metrics_limit parameter."
                        )
                        metrics_for_radar = actual_flat_metrics_in_df[:radar_metrics_limit]

                    if (
                        len(metrics_for_radar) < 3 and len(metrics_for_radar) > 0
                    ):  # Radar usually needs 3+
                        print(
                            f"Warning: Only {len(metrics_for_radar)} metrics available. Radar plot might not be informative. Generating bar plots instead."
                        )
                        for metric_to_plot in metrics_for_radar:
                            plot_metric_comparison(
                                current_scores_df,
                                metric_name=metric_to_plot,
                                aggregate_func=aggregate_func,
                                title=f"{metric_to_plot} {plot_title_suffix}",
                            )
                            viz_plt.show()
                    elif len(metrics_for_radar) >= 3:
                        plot_radar_comparison(
                            current_scores_df,
                            metrics=metrics_for_radar,
                            aggregate_func=aggregate_func,
                            title=f"Models {plot_title_suffix}",
                        )
                        viz_plt.show()
                    else:  # No metrics for radar (e.g. if radar_metrics_limit was 0 or negative)
                        print("No metrics selected for radar plot.")

        # 4. Generate CSV
        if output_csv_path:
            csv_input_threshold_results_list = []
            csv_input_gen_texts_list = []
            csv_input_ref_texts_list = []

            for model_name in self.models:
                if (
                    model_name in thresholded_results_all_models
                    and thresholded_results_all_models[model_name]
                ):
                    csv_input_threshold_results_list.append(
                        thresholded_results_all_models[model_name]
                    )
                    csv_input_gen_texts_list.append(self.llm_responses[model_name])
                    csv_input_ref_texts_list.append(self.reference_answer)

            if csv_input_threshold_results_list:
                generate_deltas_frame(
                    threshold_results=csv_input_threshold_results_list,
                    generated_texts=csv_input_gen_texts_list,
                    reference_texts=csv_input_ref_texts_list,
                    output_csv_path=output_csv_path,
                )
            else:
                print(
                    f"Warning: No non-empty thresholded results to write to CSV for path {output_csv_path}."
                )

        return current_scores_df.copy()
