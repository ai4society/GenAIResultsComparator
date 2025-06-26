from abc import ABC
from typing import Any, FrozenSet, Iterable, List, Tuple

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


class TimeSeriesDataMetric(StructuredOutputMetric, ABC):
    """
    Abstract base class for metrics evaluating time-series data,
    which can be textual or structured.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class ActionSequenceDiff(PlanningSequenceMetric):
    """
    Calculates the difference between two planning action sequences based on the
    Longest Common Subsequence (LCS). The score is normalized to [0, 1],
    where 1 indicates a perfect match.

    Input strings are expected to be comma-separated actions.
    Concurrent actions can be represented in curly braces, e.g., "a1, {a2, a3}, a4".
    """

    def __init__(self, **kwargs: Any):
        """Initialize the ActionSequenceDiff metric."""
        super().__init__(**kwargs)

    def _parse_planning_sequence(self, text_sequence: str) -> List[str | frozenset]:
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

        parsed_gen = self._parse_planning_sequence(gen_seq_str)
        parsed_ref = self._parse_planning_sequence(ref_seq_str)

        if not parsed_gen and not parsed_ref:
            return 1.0  # Both empty, perfect match

        lcs_len = self._lcs_length(parsed_gen, parsed_ref)
        max_len = max(len(parsed_gen), len(parsed_ref))

        if max_len == 0:  # Should be covered by the above, but as a safeguard
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


class TimeSeriesElementDiff(TimeSeriesDataMetric):
    """
    Calculates the difference between two time series based on the Jaccard similarity
    of their time points (keys). The values associated with time points are not
    considered in this version, focusing on structural presence/absence of time points.
    The score is normalized to [0, 1], where 1 indicates identical sets of time points.

    Input strings are expected to be comma-separated "key:value" pairs,
    e.g., "t1:70, t2:72, t3:75".
    """

    def __init__(self, **kwargs: Any):
        """Initialize the TimeSeriesElementDiff metric."""
        super().__init__(**kwargs)

    def _parse_time_series(self, text_series: str) -> List[Tuple[str, float]]:
        """
        Parses a string representation of a time series.
        Example: "t1:10, t2:15.5" -> [('t1', 10.0), ('t2', 15.5)]
        Malformed pairs or values are skipped with a warning.
        """
        if not text_series or text_series.isspace():
            return []

        parsed_series = []
        pairs = text_series.split(",")
        for pair_str in pairs:
            pair_str = pair_str.strip()
            if not pair_str:
                continue

            parts = pair_str.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip()
                if not key:  # Skip if key is empty, e.g. ":10"
                    print(f"Warning: Empty key in time series pair '{pair_str}'. Skipping.")
                    continue
                try:
                    value = float(value_str)
                    parsed_series.append((key, value))
                except ValueError:
                    print(
                        f"Warning: Could not parse value '{value_str}' for key '{key}' in time series pair '{pair_str}'. Skipping."
                    )
                    continue
            else:
                print(f"Warning: Could not parse time series pair '{pair_str}'. Skipping.")
                continue
        return parsed_series

    def _single_calculate(self, generated_text: str, reference_text: str, **kwargs: Any) -> float:
        """
        Calculate difference for a single pair of time series.
        Input texts are parsed into lists of (key, value) tuples.
        The score is the Jaccard index of the sets of keys.

        :param generated_text: The generated time series as a string.
        :type generated_text: str
        :param reference_text: The reference time series as a string.
        :type reference_text: str
        :param kwargs: Additional keyword arguments (not used here).
        :type kwargs: Any
        :return: Normalized score between 0 and 1, where 1 indicates a perfect match of time points (keys).
        :rtype: float
        """
        gen_ts_str = str(generated_text)
        ref_ts_str = str(reference_text)

        parsed_gen_ts = self._parse_time_series(gen_ts_str)
        parsed_ref_ts = self._parse_time_series(ref_ts_str)

        if not parsed_gen_ts and not parsed_ref_ts:
            return 1.0  # Both empty, perfect match

        gen_keys = set(item[0] for item in parsed_gen_ts)
        ref_keys = set(item[0] for item in parsed_ref_ts)

        intersection_len = len(gen_keys.intersection(ref_keys))
        union_len = len(gen_keys.union(ref_keys))

        # Should be covered if both empty, but handles if one parses to empty keys and other is also empty keys
        if union_len == 0:
            return 1.0

        return intersection_len / union_len

    def _batch_calculate(
        self,
        generated_texts: Iterable | np.ndarray | pd.Series,
        reference_texts: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | np.ndarray | pd.Series:
        """
        Calculate differences for a batch of time series.

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
