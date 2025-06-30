import warnings
from typing import cast

import numpy as np
import pandas as pd
import pytest

from gaico.metrics.structured import (
    PlanningJaccard,
    PlanningLCS,
    TimeSeriesElementDiff,
)


class TestPlanningLCS:
    """Test suite for the PlanningLCS metric."""

    @pytest.fixture(scope="class")
    def metric(self):
        return PlanningLCS()

    @pytest.mark.parametrize(
        "gen_seq, ref_seq, expected_score",
        [
            # Identical sequences
            ("a,b,c", "a,b,c", 1.0),
            ("a,{b,c},d", "a,{c,b},d", 1.0),  # Order in frozenset doesn't matter
            # Completely different sequences
            ("a,b,c", "x,y,z", 0.0),
            # Partially overlapping sequences
            ("a,b,c", "a,x,c", 2 / 3),  # LCS: a,c (len 2); max_len: 3
            ("a,b", "a,b,c", 2 / 3),  # LCS: a,b (len 2); max_len: 3
            ("a,b,c", "b,c", 2 / 3),  # LCS: b,c (len 2); max_len: 3
            # Sequences with concurrent actions
            ("a,{b,c},d", "a,b,d", 2 / 3),  # LCS: a,d (frozenset({b,c}) != 'b')
            ("a,{b,c},d", "a,{b},d", 2 / 3),  # LCS: a,d (frozenset({b,c}) != frozenset({b}))
            ("a,{b,c},{e,f}", "a,{c,b},{f,e}", 1.0),
            # Empty vs. non-empty
            ("a,b", "", 1.0),  # BaseMetric uses "a,b" as ref -> 1.0
            ("a,b", "", 1.0),  # 1.0 because the first element is of gen is used as ref.
            ("a", " ", 1.0),  # 1.0 because the first element is of gen is used as ref.
            # Tricky parsing cases
            ("a,,b", "a,b", 1.0),  # Extra comma
            (" a , b ", "a,b", 1.0),  # Spaces
            ("a,{,},b", "a,{},b", 1.0),  # Empty frozenset
            ("a,{b,,c},d", "a,{b,c},d", 1.0),  # Extra comma in set
            ("{a,b},c", "{b,a},c", 1.0),
            ("action1", "action1", 1.0),
            ("action1, action2", "action1", 1 / 2),  # LCS: action1 (len 1); max_len: 2
            # Example from prompt
            (
                "a_1, a_2, {a_3, a_4}, a_5",
                "a_1, a_2, a_5",
                3 / 4,
            ),  # LCS: a_1,a_2,a_5 (len 3); max_len: 4
        ],
    )
    def test_single_calculation(self, metric, gen_seq, ref_seq, expected_score):
        score = metric.calculate(gen_seq, ref_seq)
        assert isinstance(score, float)
        assert score == pytest.approx(expected_score)
        assert 0.0 <= score <= 1.0

    def test_batch_calculation_list(self, metric):
        gens = ["a,b", "x,y,z", "a,{b,c}"]
        refs = ["a,b,c", "x,y", "a,c,{b}"]
        expected_scores = [
            metric.calculate("a,b", "a,b,c"),  # 2/3
            metric.calculate("x,y,z", "x,y"),  # 2/3
            metric.calculate(
                "a,{b,c}", "a,c,{b}"
            ),  # LCS: a, {b,c} vs a,c,{b} -> a. len 1. max_len 3. -> 1/3
        ]
        scores = metric.calculate(gens, refs)
        assert isinstance(scores, list)
        assert len(scores) == len(expected_scores)
        for s, e in zip(scores, expected_scores):
            assert s == pytest.approx(e)

    def test_missing_reference_uses_first_gen_batch(self, metric):
        gens = ["a,b,c", "x,y"]  # First gen "a,b,c" becomes the reference for all
        expected_scores = [1.0, 0.0]
        scores = metric.calculate(gens, None)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)

    def test_batch_calculation_numpy(self, metric):
        gens_np = np.array(["a,b", "x,y,z"])
        refs_np = np.array(["a,b,c", "x,y"])
        expected_scores_np = np.array(
            [metric.calculate("a,b", "a,b,c"), metric.calculate("x,y,z", "x,y")],
            dtype=float,
        )
        scores = metric.calculate(gens_np, refs_np)
        assert isinstance(scores, np.ndarray)
        np.testing.assert_array_almost_equal(scores, expected_scores_np)

    def test_batch_calculation_pandas(self, metric):
        gens_pd = pd.Series(["a,b", "x,y,z"])
        refs_pd = pd.Series(["a,b,c", "x,y"])
        # Ensure index alignment for expected_scores_pd if gens_pd has a non-default index
        expected_scores_pd = pd.Series(
            [metric.calculate("a,b", "a,b,c"), metric.calculate("x,y,z", "x,y")],
            index=gens_pd.index,
            dtype=float,
        )
        scores = metric.calculate(gens_pd, refs_pd)
        assert isinstance(scores, pd.Series)
        pd.testing.assert_series_equal(scores, expected_scores_pd, check_dtype=False, atol=1e-6)

    # ** Tests for BaseMetric.calculate behavior **
    def test_calculate_broadcast_gen_str_ref_list(self, metric):
        gen = "a,b"
        refs = ["a,b,c", "a"]
        expected_scores = [
            metric.calculate("a,b", "a,b,c"),  # 2/3
            metric.calculate("a,b", "a"),  # LCS(ab,a)=1, max_len=2 -> 1/2
        ]
        scores = metric.calculate(gen, refs)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)

    def test_calculate_broadcast_gen_list_ref_str(self, metric):
        gens = ["a,b,c", "a"]
        ref = "a,b"
        expected_scores = [
            metric.calculate("a,b,c", "a,b"),  # LCS(abc,ab)=2, max_len=3 -> 2/3
            metric.calculate("a", "a,b"),  # LCS(a,ab)=1, max_len=2 -> 1/2
        ]
        scores = metric.calculate(gens, ref)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)

    def test_calculate_missing_reference_uses_first_gen_single_valid(self, metric):
        score = metric.calculate("a,b,c", None)  # Uses "a,b,c" as ref
        assert score == pytest.approx(1.0)

        score_empty_ref_str = metric.calculate("a,b,c", "")  # Uses "a,b,c" as ref
        assert score_empty_ref_str == pytest.approx(1.0)

    def test_calculate_missing_reference_gen_empty_raises_error(self, metric):
        with pytest.raises(
            ValueError,
            match="`reference` is missing and cannot be derived from `generated`",
        ):
            metric.calculate("", None)
        with pytest.raises(
            ValueError,
            match="`reference` is missing and cannot be derived from `generated`",
        ):
            metric.calculate("   ", None)
        with pytest.raises(
            ValueError,
            match="`reference` is missing and cannot be derived from `generated`",
        ):
            metric.calculate(["", "  "], None)

    def test_calculate_missing_reference_uses_first_gen_batch(self, metric):
        gens = ["a,b,c", "x,y"]  # First gen "a,b,c" becomes the reference for all
        expected_scores = [
            metric.calculate("a,b,c", "a,b,c"),  # 1.0
            metric.calculate("x,y", "a,b,c"),  # LCS(xy, abc)=0, max_len=3 -> 0.0
        ]
        scores = metric.calculate(gens, None)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)

        # Reference is effectively empty
        scores_empty_ref_list = metric.calculate(gens, [])
        assert isinstance(scores_empty_ref_list, list)
        assert scores_empty_ref_list == pytest.approx(expected_scores)


class TestPlanningJaccard:
    """Test suite for the PlanningJaccard metric."""

    @pytest.fixture(scope="class")
    def metric(self):
        return PlanningJaccard()

    @pytest.mark.parametrize(
        "gen_seq, ref_seq, expected_score",
        [
            # Identical sets
            ("a,b,c", "a,b,c", 1.0),
            ("a,b,c", "c,b,a", 1.0),  # Order doesn't matter
            ("a,{b,c}", "{c,b},a", 1.0),  # Concurrent actions flattened
            # Completely different sets
            ("a,b,c", "x,y,z", 0.0),
            # Partially overlapping sets
            ("a,b,c", "a,x,c", 2 / 4),  # I={a,c}, U={a,b,c,x}
            ("a,b", "a,b,c", 2 / 3),  # I={a,b}, U={a,b,c}
            ("a,{b,c},d", "a,b,d", 3 / 4),  # S1={a,b,c,d}, S2={a,b,d} -> I=3, U=4
            # Empty vs. non-empty
            ("a,b", "", 1.0),  # BaseMetric uses "a,b" as ref -> 1.0
            # Example from prompt
            ("a_1, a_2, {a_3, a_4}, a_5", "a_1, a_2, a_5", 3 / 5),
        ],
    )
    def test_single_calculation(self, metric, gen_seq, ref_seq, expected_score):
        score = metric.calculate(gen_seq, ref_seq)
        assert isinstance(score, float)
        assert score == pytest.approx(expected_score)
        assert 0.0 <= score <= 1.0

    def test_batch_calculation_pandas(self, metric):
        gens_pd = pd.Series(["a,b", "x,y"])
        refs_pd = pd.Series(["a,b,c", "a,b"])
        expected_scores_pd = pd.Series([2 / 3, 0 / 4], index=gens_pd.index, dtype=float)
        scores = metric.calculate(gens_pd, refs_pd)
        assert isinstance(scores, pd.Series)
        pd.testing.assert_series_equal(scores, expected_scores_pd, check_dtype=False, atol=1e-6)

    def test_missing_reference_uses_first_gen_batch(self, metric):
        gens = ["a,b,c", "a,x"]  # First gen "a,b,c" becomes the reference
        expected_scores = [1.0, 1 / 4]  # I={a}, U={a,b,c,x}
        scores = metric.calculate(gens, None)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)


class TestTimeSeriesElementDiff:
    """Test suite for the TimeSeriesElementDiff metric."""

    def test_init_weights(self):
        metric_default = TimeSeriesElementDiff()
        assert metric_default.key_weight == 2.0
        assert metric_default.value_weight == 1.0

        metric_custom = TimeSeriesElementDiff(key_to_value_weight_ratio=5.0)
        assert metric_custom.key_weight == 5.0
        assert metric_custom.value_weight == 1.0

        with pytest.raises(ValueError):
            TimeSeriesElementDiff(key_to_value_weight_ratio=0)
        with pytest.raises(ValueError):
            TimeSeriesElementDiff(key_to_value_weight_ratio=-1)

    @pytest.mark.parametrize(
        "gen_ts, ref_ts, expected_score",
        [
            # Identical series
            ("t1:10, t2:20", "t1:10, t2:20", 1.0),
            # Identical keys, different values
            ("t1:10", "t1:20", (2 + 1 * 0.5) / 3),  # 0.8333
            # Partially overlapping keys
            ("t1:10, t2:20", "t1:10, t3:30", (2 + 1 * 1.0) / 9),  # 0.3333
            # Completely different keys
            ("t1:10", "t2:20", 0.0),
            # Empty vs. non-empty
            ("t1:10", "", 1.0),  # BaseMetric uses "t1:10" as ref -> 1.0
            # Cases with zero values
            ("t1:10", "t1:0", (2 + 1 * 0.0) / 3),  # 0.6666
            ("t1:0", "t1:0", (2 + 1 * 1.0) / 3),  # 1.0
            # Example from prompt
            ("t_1: 70, t_2: 72, t_3: 75", "t_1: 70, t_3: 70", 5.92857 / 9),  # 0.6587
        ],
    )
    def test_single_calculation_default_weight(self, gen_ts, ref_ts, expected_score):
        metric = TimeSeriesElementDiff()
        score = metric.calculate(gen_ts, ref_ts)
        assert isinstance(score, float)
        assert score == pytest.approx(expected_score)
        assert 0.0 <= score <= 1.0

    def test_single_calculation_custom_weight(self):
        metric = TimeSeriesElementDiff(key_to_value_weight_ratio=4.0)
        # Test case: ("t1:10", "t1:20")
        # key_weight=4, value_weight=1. max_score = 1*(4+1)=5
        # value_sim = 0.5. total_score = 4 + 1*0.5 = 4.5
        # final_score = 4.5 / 5 = 0.9
        score = metric.calculate("t1:10", "t1:20")
        score = cast(float, score)
        assert score == pytest.approx(0.9)

    def test_parsing_robustness_and_warnings(self):
        metric = TimeSeriesElementDiff()
        # Test that malformed pairs and bad values are skipped with warnings,
        # but the calculation still proceeds with the valid parts.
        with pytest.warns(UserWarning) as record:
            # "t2:bad" will be skipped, ":30" will be treated as unkeyed.
            # gen_dict becomes {'t1': [10.0], '_UNKEYED_': [30.0]}
            # ref_dict becomes {'t1': [10.0], '_UNKEYED_': []}
            score = metric.calculate("t1:10, t2:bad, :30", "t1:10")

        messages = [str(w.message) for w in record]
        assert len(messages) == 2
        assert any("Could not parse value" in m for m in messages)
        assert any("Empty key in time series pair" in m for m in messages)

        # Calculation should still work.
        # Keyed: t1 vs t1 -> perfect match. Score = 1.0. Num items = 1.
        # Unkeyed: [30.0] vs [] -> Jaccard = 0. Score = 0.0. Num items = 0.
        # Final score = (1.0 * 1 + 0.0 * 0) / 1 = 1.0
        assert isinstance(score, float)
        assert score == pytest.approx(1.0)

    def test_parsing_duplicate_keys(self):
        metric = TimeSeriesElementDiff()
        # The new parser should handle duplicate keys by creating a list of values.
        # No warning should be issued as this is now supported behavior.
        with warnings.catch_warnings(record=True) as w:
            gen_ts = "t1:10, t1:20"
            ref_ts = "t1:10"

            # The parser should produce {'t1': [10.0, 20.0]}
            # The metric logic compares only the first element, so it compares 10 vs 10.
            score = metric.calculate(gen_ts, ref_ts)
            assert isinstance(score, float)

            # Assert that no "Duplicate key" warning was raised
            assert len(w) == 0

            # Keyed: t1 vs t1. Compares 10 vs 10. Score = 1.0. Num items = 1.
            # Unkeyed: [] vs []. Score = 1.0 (or 0 if weighted by 0 items). Num items = 0.
            # Final score = (1.0 * 1 + 1.0 * 0) / 1 = 1.0
            assert score == pytest.approx(1.0)

    def test_batch_calculation_numpy(self):
        metric = TimeSeriesElementDiff()
        gens_np = np.array(["t1:10", "t1:10, t2:20"])
        refs_np = np.array(["t1:20", "t1:10, t3:30"])
        expected_scores_np = np.array(
            [
                (2 + 1 * 0.5) / 3,  # 0.8333
                (2 + 1 * 1.0) / 9,  # 0.3333
            ],
            dtype=float,
        )
        scores = metric.calculate(gens_np, refs_np)
        assert isinstance(scores, np.ndarray)
        np.testing.assert_array_almost_equal(scores, expected_scores_np)

    def test_missing_reference_uses_first_gen_batch(self):
        metric = TimeSeriesElementDiff()
        gens = ["t1:10, t2:20", "t1:10"]  # First gen becomes ref
        # 1. "t1:10, t2:20" vs "t1:10, t2:20" -> 1.0
        # 2. "t1:10" vs "t1:10, t2:20" -> I={t1}, U={t1,t2}. max=6. score=3. -> 0.5
        expected_scores = [1.0, 0.5]
        scores = metric.calculate(gens, None)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)
