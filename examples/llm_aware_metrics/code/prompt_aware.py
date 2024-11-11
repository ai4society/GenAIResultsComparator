from typing import Any, Dict, List, Optional, Union

from llm_metrics.base import BaseMetric

from .base import LLMAwareMetric


class PromptAwareMetric(LLMAwareMetric):
    """
    A metric that combines the prompt with the text before calculating the similarity.
    """

    def __init__(self, base_metric: BaseMetric):
        """
        Initialize the metric with a base metric.
        :param base_metric: Any metric class from llm_metrics, such as BLEU, ROUGE, BERTScore, etc. Must have a `calculate` and `batch_calculate` method.
        """
        # Check if base_metric has the calculate and batch_calculate methods
        if not callable(getattr(base_metric, "calculate", None)):
            raise ValueError("Base metric must have a `calculate` method")
        if not callable(getattr(base_metric, "batch_calculate", None)):
            raise ValueError("Base metric must have a `batch_calculate` method")

        self.base_metric = base_metric

    def calculate_with_prompt(
        self,
        text1: str,
        text2: str,
        prompt1: str,
        prompt2: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Calculate similarity considering prompts and metadata.

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

        # Create context-aware texts
        full_text1 = f"{prompt1} {text1}"
        full_text2 = f"{prompt2} {text2}"

        return self.base_metric.calculate(full_text1, full_text2)

    def batch_calculate_with_prompt(
        self,
        texts1: List[str],
        texts2: List[str],
        prompts1: List[str],
        prompts2: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[List[float], List[Dict[str, float]]]:
        """
        Calculate similarity for batches of texts with prompts.

        :param texts1: A list of the first responses
        :param texts2: A list of the second responses
        :param prompts1: A list of the first prompts
        :param prompts2: An optional list of the second prompts, defaults to None
        :param metadata: Optional metadata
        :return: A list of the combined scores
        :rtype: Union[List[float], List[Dict[str, float]]]
        """
        prompts2 = prompts2 or prompts1
        full_texts1 = [f"{p} {t}" for p, t in zip(prompts1, texts1)]
        full_texts2 = [f"{p} {t}" for p, t in zip(prompts2, texts2)]

        return self.base_metric.batch_calculate(full_texts1, full_texts2)
