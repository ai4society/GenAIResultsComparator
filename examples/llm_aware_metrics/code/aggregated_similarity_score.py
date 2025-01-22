from typing import Any, Dict, List, Optional, Union

from llm_metrics.base import BaseMetric

from .base import LLMAwareMetric


def _get_combined_score(a: float, b: float, c: float) -> float:
    """
    A simple function to combine three scores.
    :param a: First score
    :type a: float
    :param b: Second score
    :type b: float
    :param c: Third score
    :type c: float
    :return: Combined score
    :rtype: float
    """
    return (a + b + c) / 3


class AggregatedSimilarityMetric(LLMAwareMetric):
    """
    A metric that combines a base metric with an aggregated similarity metric.
    The aggregated similarity metric is used to calculate the similarity between two responses.
    The base metric is used to calculate the similarity between the two responses.
    """

    def __init__(self, base_metric: BaseMetric, aggregated_metric: Optional[BaseMetric] = None):
        """
        Initialize the metric with a base metric and an optional aggregated similarity  metric.

        :param base_metric: Primary metric to use for comparison.
        :param aggregated_metric: Metric for measuring aggregated similarity, (if None, uses base_metric)
        """
        self.base_metric = base_metric

        # Check if base metric has the calculate method
        if not callable(getattr(base_metric, "calculate", None)):
            raise ValueError("Base metric must have a `calculate` method")

        # Check if aggregated metric has the calculate method
        if aggregated_metric is not None and not callable(
            getattr(aggregated_metric, "calculate", None)
        ):
            raise ValueError("Alignment metric must have a `calculate` method")

        self.aggregated_metric = aggregated_metric or base_metric

    def calculate_prompt_aggregation(
        self, prompt: str, response: str
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate how similar the response are to the prompt.
        Uses the aggregated metric to calculate the aggregation score.
        """
        return self.aggregated_metric.calculate(prompt, response)

    def calculate_with_prompt(
        self,
        text1: str,
        text2: str,
        prompt1: str,
        prompt2: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate the combined score of aggregated and response similarity.
        The method works by calculating the aggregated similarity score for both prompts and responses respectively.
        Then it calculates the response similarity score of the two responses using the base metric.
        Finally, it combines the scores to get the final score.
        The score combination can be done in various ways depending on the use case. See the `_get_combined_score` function.

        :param text1: The first response
        :type text1: str
        :param text2: The second response
        :type text2: str
        :param prompt1: The first prompt
        :type prompt1: str
        :param prompt2: The optional second prompt, defaults to None
        :type prompt2: str, optional
        :param metadata: Optional metadata
        :type metadata: Dict[str, Any], optional
        :return: The combined score
        :rtype: Union[float, Dict[str, float]]
        """
        prompt2 = prompt2 or prompt1

        # Calculate prompt-response aggregation scores
        aggregation1: Union[float, Dict[str, float]] = self.calculate_prompt_aggregation(
            prompt1, text1
        )
        aggregation2: Union[float, Dict[str, float]] = self.calculate_prompt_aggregation(
            prompt2, text2
        )

        # Calculate response similarity
        response_similarity: Union[float, Dict[str, float]] = self.base_metric.calculate(
            text1, text2
        )

        # Combine scores (can be weighted differently based on requirements)...

        # Using a BaseMetric class might output, either a single float or a
        # dictionary with multiple scores, such as 'precision', 'recall', 'f1'
        if (
            isinstance(response_similarity, float)
            and isinstance(aggregation1, float)
            and isinstance(aggregation2, float)
        ):
            return _get_combined_score(aggregation1, aggregation2, response_similarity)

        # If it is a dictionary, then combine each score separately
        elif (
            isinstance(response_similarity, dict)
            and isinstance(aggregation1, dict)
            and isinstance(aggregation2, dict)
        ):
            combined_scores = {}
            for (
                key
            ) in response_similarity:  # Assuming that the keys are the same for all dictionaries
                combined_scores[key] = _get_combined_score(
                    aggregation1.get(key, 0.0),
                    aggregation2.get(key, 0.0),
                    response_similarity.get(key, 0.0),
                )
            return combined_scores

        else:
            raise ValueError("Incompatible return types for aggregation and response similarity")

    def batch_calculate_with_prompt(
        self,
        texts1: List[str],
        texts2: List[str],
        prompts1: List[str],
        prompts2: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[float | Dict[str, float]]:
        """
        Calculate the combined score of prompt aggregation and response similarity for a batch of responses.
        The method works by calculating the prompt aggregation score for both prompts and responses respectively.
        Then it calculates the response similarity score of the two responses using the base metric.
        Finally, it combines the scores to get the final score.
        Currently, the scores are combined by taking the average of the three scores.

        :param texts1: A list of the first responses
        :param texts2: A list of the second responses
        :param prompts1: A list of the first prompts
        :param prompts2: An optional list of the second prompts, defaults to None
        :param metadata: Optional metadata
        :return: A list of the combined scores
        :rtype: Union[List[float], List[Dict[str, float]]]
        """
        prompts2 = prompts2 or prompts1
        results = []

        for t1, t2, p1, p2 in zip(texts1, texts2, prompts1, prompts2):
            results.append(self.calculate_with_prompt(t1, t2, p1, p2, metadata))

        return results
