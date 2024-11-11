from typing import Any, Dict, List, Optional, Union

from llm_metrics.base import BaseMetric

from .base import LLMAwareMetric


def get_combined_score(a: float, b: float, c: float) -> float:
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


class PromptAlignmentMetric(LLMAwareMetric):
    """
    A metric that combines a base metric with a prompt alignment metric.
    The prompt alignment metric is used to calculate how well the response aligns with the prompt.
    The base metric is used to calculate the similarity between the two responses.
    """

    def __init__(self, base_metric: BaseMetric, alignment_metric: Optional[BaseMetric] = None):
        """
        Initialize the metric with a base metric and an optional alignment metric.

        :param base_metric: Primary metric to use for comparison.
        :param alignment_metric: Metric for measuring prompt alignment (if None, uses base_metric)
        """
        self.base_metric = base_metric

        # Check if base metric has the calculate method
        if not callable(getattr(base_metric, "calculate", None)):
            raise ValueError("Base metric must have a `calculate` method")

        # Check if alignment metric has the calculate method
        if alignment_metric is not None and not callable(
            getattr(alignment_metric, "calculate", None)
        ):
            raise ValueError("Alignment metric must have a `calculate` method")

        self.alignment_metric = alignment_metric or base_metric

    def calculate_prompt_alignment(
        self, prompt: str, response: str
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate how well the response aligns with the prompt.
        Used the alignment metric to calculate the alignment score.
        """
        return self.alignment_metric.calculate(prompt, response)

    def calculate_with_prompt(
        self,
        text1: str,
        text2: str,
        prompt1: str,
        prompt2: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate the combined score of prompt alignment and response similarity.
        The method works by calculating the prompt alignment score for both prompts and responses respectively.
        Then it calculates the response similarity score of the two responses using the base metric.
        Finally, it combines the scores to get the final score.
        Currently, the scores are combined by taking the average of the three scores.

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

        # Calculate prompt-response alignment scores
        alignment1: Union[float, Dict[str, float]] = self.calculate_prompt_alignment(prompt1, text1)
        alignment2: Union[float, Dict[str, float]] = self.calculate_prompt_alignment(prompt2, text2)

        # Calculate response similarity
        response_similarity: Union[float, Dict[str, float]] = self.base_metric.calculate(
            text1, text2
        )

        # Combine scores (can be weighted differently based on requirements)...

        # Using a BaseMetric class might output, either a single float or a
        # dictionary with multiple scores, such as 'precision', 'recall', 'f1'
        if isinstance(response_similarity, float):
            return get_combined_score(alignment1, alignment2, response_similarity)

        elif isinstance(
            response_similarity, dict
        ):  # If it is a dictionary, then combine each score separately
            combined_scores = {}
            for (
                key
            ) in response_similarity:  # Assuming that the keys are the same for all dictionaries
                combined_scores[key] = get_combined_score(
                    alignment1.get(key, 0.0),
                    alignment2.get(key, 0.0),
                    response_similarity.get(key, 0.0),
                )
            return combined_scores

    def batch_calculate_with_prompt(
        self,
        texts1: List[str],
        texts2: List[str],
        prompts1: List[str],
        prompts2: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[List[float], List[Dict[str, float]]]:
        """
        Calculate the combined score of prompt alignment and response similarity for a batch of responses.
        The method works by calculating the prompt alignment score for both prompts and responses respectively.
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
