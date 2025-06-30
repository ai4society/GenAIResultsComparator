import warnings
from abc import ABC
from typing import Any, Dict, FrozenSet, Iterable, List, Set

import numpy as np
import pandas as pd

from ..base import BaseMetric


class StructuredOutputMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate on structured text or sequences.
    Input is typically a parsed representation of the sequence or structure.
    """

    # Ensure __init__ is present if BaseMetric or other parents require it.
    def __init__(self, **kwargs: Any):
        super().__init__()  # Call super() in case BaseMetric's hierarchy changes


class PlanningSequenceMetric(StructuredOutputMetric, ABC):
    """
    Abstract base class for metrics designed to evaluate action sequences,
    often found in automated planning outputs.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @staticmethod
    def _parse_planning_sequence(text_sequence: str) -> List[str | frozenset]:
        """
        Parses a string representation of a planning sequence.
        Example: "a1, {a2, a3}, a4" -> ['a1', frozenset({'a2', 'a3'}), 'a4']

        :param text_sequence: The input string representing the action sequence.
        :type text_sequence: str
        :return: A list of actions or frozensets of concurrent actions.
        :rtype: List[str | frozenset]
        """
        if not text_sequence or text_sequence.isspace():
            return []

        items_raw_strings = []
        current_item_buffer = ""
        in_set_scope = 0
        for char in text_sequence:
            if char == "{":
                in_set_scope += 1
                current_item_buffer += char
            elif char == "}":
                in_set_scope -= 1
                current_item_buffer += char
            elif char == "," and in_set_scope == 0:
                items_raw_strings.append(current_item_buffer.strip())
                current_item_buffer = ""
            else:
                current_item_buffer += char
        items_raw_strings.append(current_item_buffer.strip())

        # Filter out empty strings that might result from "a,,b" or trailing/leading commas
        items_raw_strings = [s for s in items_raw_strings if s]

        parsed_items: List[str | FrozenSet] = []
        for item_str in items_raw_strings:
            if item_str.startswith("{") and item_str.endswith("}"):
                content_str = item_str[1:-1].strip()
                if not content_str:  # Handles "{}"
                    parsed_items.append(frozenset())
                else:
                    actions_in_set = [s.strip() for s in content_str.split(",")]
                    actions_in_set_filtered = [s for s in actions_in_set if s]
                    if actions_in_set_filtered:
                        parsed_items.append(frozenset(actions_in_set_filtered))
                    else:  # Handles "{ , }" or similar resulting in no valid actions
                        parsed_items.append(frozenset())
            else:
                parsed_items.append(item_str)
        return parsed_items


class TimeSeriesDataMetric(StructuredOutputMetric, ABC):
    """
    Abstract base class for metrics evaluating time-series data,
    which can be textual or structured.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class PlanningLCS(PlanningSequenceMetric):
    """
    Calculates the difference between two planning action sequences based on the
    Longest Common Subsequence (LCS). The score is normalized to [0, 1],
    where 1 indicates a perfect match. This metric respects the order of actions.

    Input strings are expected to be comma-separated actions.
    Concurrent actions can be represented in curly braces, e.g., "a1, {a2, a3}, a4".
    """

    def __init__(self, **kwargs: Any):
        """Initialize the PlanningLCS metric."""
        super().__init__(**kwargs)

    def _lcs_length(self, seq1: List[Any], seq2: List[Any]) -> int:
        """
        Computes the length of the Longest Common Subsequence.

        :param seq1: First sequence of actions.
        :type seq1: List[Any]
        :param seq2: Second sequence of actions.
        :type seq2: List[Any]
        :return: Length of the longest common subsequence.
        :rtype: int
        """
        m = len(seq1)
        n = len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def _single_calculate(self, generated_text: str, reference_text: str, **kwargs: Any) -> float:
        """
        Calculate difference for a single pair of action sequences.
        Input texts are parsed into sequences of actions/action-sets.
        The score is based on the Longest Common Subsequence (LCS) length,
        normalized by the maximum possible length.

        :param generated_text: The generated action sequence as a string.
        :type generated_text: str
        :param reference_text: The reference action sequence as a string.
        :type reference_text: str
        :param kwargs: Additional keyword arguments (not used here).
        :type kwargs: Any
        :return: Normalized score between 0 and 1, where 1 indicates a perfect match.
        :rtype: float
        """
        # Type casting, as BaseMetric passes Any, but we expect strings for parsing
        gen_seq_str = str(generated_text)
        ref_seq_str = str(reference_text)

        parsed_gen = PlanningSequenceMetric._parse_planning_sequence(gen_seq_str)
        parsed_ref = PlanningSequenceMetric._parse_planning_sequence(ref_seq_str)

        if not parsed_gen and not parsed_ref:
            return 1.0  # Both empty, perfect match

        lcs_len = self._lcs_length(parsed_gen, parsed_ref)
        max_len = max(len(parsed_gen), len(parsed_ref))

        if max_len == 0:
            return 1.0

        return lcs_len / max_len

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | np.ndarray | pd.Series:
        """
        Calculate differences for a batch of action sequences.

        This method processes pairs of generated and reference sequences, applying the _single_calculate method to each pair.

        :param generated_texts: Iterable of generated action sequences.
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Iterable of reference action sequences.
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments (not used here).
        :type kwargs: Any
        :return: List of normalized scores for each pair, or a numpy array or pandas Series if the input is of those types.
        :rtype: List[float] | np.ndarray | pd.Series
        """
        results = [
            self._single_calculate(str(gen), str(ref), **kwargs)
            for gen, ref in zip(generated_texts, reference_texts)
        ]

        if isinstance(generated_texts, np.ndarray):
            return np.array(results, dtype=float)
        if isinstance(generated_texts, pd.Series):
            return pd.Series(results, index=generated_texts.index, dtype=float)
        return results


class PlanningJaccard(PlanningSequenceMetric):
    """
    Calculates the Jaccard similarity between the sets of actions from two
    planning sequences. The score is normalized to [0, 1], where 1 indicates
    that both sequences contain the exact same set of actions, ignoring order
    and frequency.

    Concurrent actions are flattened into the set.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the PlanningJaccard metric."""
        super().__init__(**kwargs)

    def _flatten_sequence_to_set(self, parsed_sequence: List[Any]) -> set:
        """Converts a parsed sequence into a flat set of unique actions."""
        flat_set: Set = set()
        for item in parsed_sequence:
            if isinstance(item, frozenset):
                flat_set.update(item)
            else:
                flat_set.add(item)
        return flat_set

    def _single_calculate(self, generated_text: str, reference_text: str, **kwargs: Any) -> float:
        """Calculate Jaccard similarity for a single pair of action sequences."""
        gen_seq_str = str(generated_text)
        ref_seq_str = str(reference_text)

        parsed_gen = PlanningSequenceMetric._parse_planning_sequence(gen_seq_str)
        parsed_ref = PlanningSequenceMetric._parse_planning_sequence(ref_seq_str)

        set_gen = self._flatten_sequence_to_set(parsed_gen)
        set_ref = self._flatten_sequence_to_set(parsed_ref)

        if not set_gen and not set_ref:
            return 1.0

        intersection = set_gen.intersection(set_ref)
        union = set_gen.union(set_ref)

        if not union:
            return 1.0

        return len(intersection) / len(union)

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | np.ndarray | pd.Series:
        """Calculate Jaccard similarities for a batch of action sequences."""
        results = [
            self._single_calculate(str(gen), str(ref), **kwargs)
            for gen, ref in zip(generated_texts, reference_texts)
        ]

        if isinstance(generated_texts, np.ndarray):
            return np.array(results, dtype=float)
        if isinstance(generated_texts, pd.Series):
            return pd.Series(results, index=generated_texts.index, dtype=float)
        return results


class TimeSeriesElementDiff(TimeSeriesDataMetric):
    """
    Calculates a weighted difference between two time series.
    This metric considers both the presence of time points (keys) and the
    similarity of their corresponding values. It assigns a higher weight to
    matching keys than to matching values.

    The final score is normalized to [0, 1], where 1 indicates a perfect match.

    Input strings are expected to be comma-separated "key:value" pairs,
    e.g., "t1:70, t2:72, t3:75".
    """

    def __init__(self, key_to_value_weight_ratio: float = 2.0, **kwargs: Any):
        """
        Initialize the TimeSeriesElementDiff metric.

        :param key_to_value_weight_ratio: The weight of a key match relative to a perfect value match.
                                          For example, a ratio of 2 means a key match is worth twice as much
                                          as a value match. Defaults to 2.0.
        :type key_to_value_weight_ratio: float
        """
        super().__init__(**kwargs)
        if key_to_value_weight_ratio <= 0:
            raise ValueError("key_to_value_weight_ratio must be positive.")
        self.key_weight = key_to_value_weight_ratio
        self.value_weight = 1.0

    def _parse_time_series(self, text_series: str) -> Dict[str, List[float]]:
        """
        Parses a string representation of a time series into a dictionary.
        Handles both keyed ("k:v") and unkeyed ("v") values. Unkeyed values
        are collected under a special '_UNKEYED_' key.
        Example: "t1:10, 15.5, t2:20" -> {'t1': [10.0], 't2': [20.0], '_UNKEYED_': [15.5]}
        """
        if not text_series or text_series.isspace():
            return {}

        parsed_dict: Dict[str, List[float]] = {"_UNKEYED_": []}
        pairs = text_series.split(",")
        for pair_str in pairs:
            pair_str = pair_str.strip()
            if not pair_str:
                continue

            try:
                parts = pair_str.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    value = float(value_str)  # Parse value BEFORE modifying dict

                    if not key:
                        warnings.warn(
                            f"Warning: Empty key in time series pair '{pair_str}'. Treating as unkeyed."
                        )
                        parsed_dict["_UNKEYED_"].append(value)
                    else:
                        if key not in parsed_dict:
                            parsed_dict[key] = []
                        parsed_dict[key].append(value)
                else:
                    # This is an unkeyed value
                    value_str = parts[0].strip()
                    value = float(value_str)  # Parse value BEFORE modifying dict
                    parsed_dict["_UNKEYED_"].append(value)
            except ValueError:
                warnings.warn(
                    f"Warning: Could not parse value in time series pair '{pair_str}'. Skipping."
                )
        return parsed_dict

    def _single_calculate(self, generated_text: str, reference_text: str, **kwargs: Any) -> float:
        """
        Calculate a weighted difference for a single pair of time series,
        handling both keyed and unkeyed values robustly.

        :param generated_text: The generated time series as a string.
        :type generated_text: str
        :param reference_text: The reference time series as a string.
        :type reference_text: str
        :param kwargs: Additional keyword arguments (not used here).
        :type kwargs: Any
        :return: Normalized score between 0 and 1, where 1 indicates a perfect match of time points (keys).
        :rtype: float
        """
        gen_dict = self._parse_time_series(str(generated_text))
        ref_dict = self._parse_time_series(str(reference_text))

        if not any(gen_dict.values()) and not any(ref_dict.values()):
            return 1.0

        # ** 1. Keyed Value Comparison **
        keyed_gen_keys = set(gen_dict.keys()) - {"_UNKEYED_"}
        keyed_ref_keys = set(ref_dict.keys()) - {"_UNKEYED_"}
        all_keyed_keys = keyed_gen_keys.union(keyed_ref_keys)

        keyed_score = 0.0
        if all_keyed_keys:
            total_keyed_score = 0.0
            max_possible_keyed_score = 0.0
            for key in all_keyed_keys:
                # For simplicity, compare the first value if a key has multiple.
                v_gen = gen_dict.get(key, [0.0])[0]
                v_ref = ref_dict.get(key, [0.0])[0]

                max_possible_keyed_score += self.key_weight + self.value_weight

                if key in keyed_gen_keys and key in keyed_ref_keys:
                    total_keyed_score += self.key_weight
                    denominator = abs(v_ref)
                    value_sim = (
                        (1.0 - abs(v_gen - v_ref) / denominator)
                        if denominator != 0
                        else (1.0 if v_gen == 0 else 0.0)
                    )
                    total_keyed_score += self.value_weight * max(0.0, value_sim)

            keyed_score = (
                total_keyed_score / max_possible_keyed_score
                if max_possible_keyed_score > 0
                else 1.0
            )

        # ** 2. Unkeyed Value Comparison (using Jaccard) **
        unkeyed_gen = set(gen_dict.get("_UNKEYED_", []))
        unkeyed_ref = set(ref_dict.get("_UNKEYED_", []))

        unkeyed_score = 0.0
        if unkeyed_ref or unkeyed_gen:
            intersection = len(unkeyed_gen.intersection(unkeyed_ref))
            union = len(unkeyed_gen.union(unkeyed_ref))
            unkeyed_score = intersection / union if union > 0 else 1.0

        # ** 3. Combine Scores via Weighted Average **
        # Weight by the number of items in the reference
        num_keyed_items = len(keyed_ref_keys)
        num_unkeyed_items = len(unkeyed_ref)
        total_items = num_keyed_items + num_unkeyed_items

        if total_items == 0:
            # If reference is empty, score is 1 only if generated is also empty.
            return 1.0 if not (keyed_gen_keys or unkeyed_gen) else 0.0

        return (keyed_score * num_keyed_items + unkeyed_score * num_unkeyed_items) / total_items

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | np.ndarray | pd.Series:
        """
        Calculate weighted differences for a batch of time series.

        :param generated_texts: Iterable of generated time series.
        :type generated_texts: Iterable | np.ndarray | pd.Series
        :param reference_texts: Iterable of reference time series.
        :type reference_texts: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments (not used here).
        :type kwargs: Any
        :return: List of normalized scores for each pair, or a numpy array or pandas Series if the input is of those types.
        :rtype: List[float] | np.ndarray | pd.Series
        """
        results = [
            self._single_calculate(str(gen), str(ref), **kwargs)
            for gen, ref in zip(generated_texts, reference_texts)
        ]

        if isinstance(generated_texts, np.ndarray):
            return np.array(results, dtype=float)
        if isinstance(generated_texts, pd.Series):
            return pd.Series(results, index=generated_texts.index, dtype=float)
        return results
