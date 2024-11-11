import json
from typing import Any, Dict, List, Optional
from warnings import warn

from jsonschema import ValidationError, validate

from llm_metrics.base import BaseMetric

from .base import LLMAwareMetric


def validate_schema(text: str, schema: Dict) -> float:
    """
    Validate a text against a JSON schema.

    :param text: Text to validate
    :type text: str
    :param schema:  JSON schema to validate against
    :type schema: Dict
    :return: 1.0 if the text is valid, 0.0 otherwise
    :rtype: float
    """
    try:
        # Try to parse the text as JSON
        json_data = json.loads(text)
        # Validate against schema
        validate(instance=json_data, schema=schema)
        return 1.0
    except (json.JSONDecodeError, ValidationError):
        return 0.0


class SchemaAwareMetric(LLMAwareMetric):
    """
    A metric that combines schema validation with prompt-aware comparison.
    Currently, it uses a simple combination of schema validation and prompt-aware comparison.
    The schema validation is done using JSON schema. If the text does not match the schema, the score is 0.0.
    The prompt-aware comparison is done using a base metric and combining the scores of the prompt and the text (concatenated).
    """

    def __init__(self, base_metric: BaseMetric):
        """
        Initialize the metric with a base metric.
        :param base_metric: Any metric class from llm_metrics, such as BLEU, ROUGE, BERTScore, etc. Must have a `calculate` method.
        """
        # Check if base_metric has the calculate method
        if not callable(getattr(base_metric, "calculate", None)):
            raise ValueError("Base metric must have a `calculate` method")

        self.base_metric = base_metric

    def _calculate_combined_score(
        self, text1: str, text2: str, prompt1: str, prompt2: str, schema: Dict
    ) -> float:
        """
        Calculate a combined score that considers both schema validation
        and prompt-aware comparison.

        :param text1: First text to compare
        :param text2: Second text to compare
        :param prompt1: Prompt for first text
        :param prompt2: Prompt for second text
        :param schema: JSON schema for validation
        :return: Combined score between 0 and 1
        """
        # Validate schema
        schema_score1 = validate_schema(text1, schema)
        schema_score2 = validate_schema(text2, schema)

        # If either text doesn't match schema, return 0
        if schema_score1 == 0 or schema_score2 == 0:
            warn("One or more outputs do not match schema. Returning 0.0.")
            return 0.0

        # Calculate prompt-aware similarity
        # Include prompts in the comparison to ensure context is considered
        full_text1 = f"{prompt1} {text1}"
        full_text2 = f"{prompt2} {text2}"
        similarity_score = self.base_metric.calculate(full_text1, full_text2)

        # Combine scores
        # Weights could be adjusted based on your needs
        SCHEMA_WEIGHT, SIMILARITY_WEIGHT = 0.4, 0.6

        combined_score = (
            SCHEMA_WEIGHT * ((schema_score1 + schema_score2) / 2)
            + SIMILARITY_WEIGHT * similarity_score
        )

        return combined_score

    def calculate_with_prompt(
        self,
        text1: str,
        text2: str,
        prompt1: str,
        prompt2: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate similarity considering prompts and schema validation.

        :param text1: First text to compare
        :param text2: Second text to compare
        :param prompt1: Prompt for first text
        :param prompt2: Optional prompt for second text (defaults to prompt1)
        :param metadata: Must contain 'schema' key with JSON schema
        :return: Similarity score between 0 and 1 where 1 is most similar
        """
        if not metadata or "schema" not in metadata:
            raise ValueError("Schema must be provided in metadata.")

        prompt2 = prompt2 or prompt1
        return self._calculate_combined_score(text1, text2, prompt1, prompt2, metadata["schema"])

    def batch_calculate_with_prompt(
        self,
        texts1: List[str],
        texts2: List[str],
        prompts1: List[str],
        prompts2: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Calculate similarity for batches of texts with prompts and schema validation.

        :param texts1: List of first texts to compare
        :param texts2: List of second texts to compare
        :param prompts1: List of prompts for first texts
        :param prompts2: Optional list of prompts for second texts
        :param metadata: Must contain 'schema' key with JSON schema
        :return: List of similarity scores between 0 and 1
        """
        if not metadata or "schema" not in metadata:
            raise ValueError("Schema must be provided in metadata.")

        prompts2 = prompts2 or prompts1

        return [
            self._calculate_combined_score(t1, t2, p1, p2, metadata["schema"])
            for t1, t2, p1, p2 in zip(texts1, texts2, prompts1, prompts2)
        ]
