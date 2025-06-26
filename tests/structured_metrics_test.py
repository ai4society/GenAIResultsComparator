import numpy as np
import pandas as pd
import pytest

from gaico.metrics.structured import ActionSequenceDiff, TimeSeriesElementDiff


class TestActionSequenceDiff:
    @pytest.fixture(scope="class")
    def metric(self):
        return ActionSequenceDiff()

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
            match="generated_texts cannot consist solely of empty or whitespace strings if reference is to be derived",
        ):
            metric.calculate("", None)
        with pytest.raises(
            ValueError,
            match="generated_texts cannot consist solely of empty or whitespace strings if reference is to be derived",
        ):
            metric.calculate("   ", None)
        with pytest.raises(
            ValueError,
            match="generated_texts cannot consist solely of empty or whitespace strings if reference is to be derived",
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

        scores_empty_ref_list_str = metric.calculate(gens, ["", "   "])
        assert isinstance(scores_empty_ref_list_str, list)
        assert scores_empty_ref_list_str == pytest.approx(expected_scores)


class TestTimeSeriesElementDiff:
    @pytest.fixture(scope="class")
    def metric(self):
        return TimeSeriesElementDiff()

    @pytest.mark.parametrize(
        "gen_ts, ref_ts, expected_score",
        [
            # Identical time series (keys)
            ("t1:10,t2:20", "t1:100,t2:200", 1.0),  # Values differ, keys same
            ("t1:10, t2:20", "t2:200, t1:100", 1.0),  # Order doesn't matter
            # Completely different time series (keys)
            ("t1:10,t2:20", "t3:30,t4:40", 0.0),  # Intersection 0, Union 4
            # Partially overlapping time series (keys)
            ("t1:10,t2:20,t3:30", "t1:1,t3:3,t4:4", 0.5),  # I={t1,t3} (2), U={t1,t2,t3,t4} (4)
            ("t1:10,t2:20", "t1:1,t2:2,t3:3", 2 / 3),  # I={t1,t2} (2), U={t1,t2,t3} (3)
            # Empty vs. non-empty
            ("t1:10", "", 1.0),  # 1.0 because the first element is of gen is used as ref.
            ("t1:10", " ", 1.0),  # 1.0 because the first element is of gen is used as ref.
            # Tricky parsing / duplicate keys
            ("t1:10,t2:bad,t3:30", "t1:1,t3:3", 1.0),  # t2:bad skipped. Gen keys {t1,t3}
            ("t1:10,,t2:20", "t1:1,t2:2", 1.0),  # Extra comma. Gen keys {t1,t2}
            (":10,t1:20", "t1:1", 1.0),  # :10 skipped. Gen keys {t1}
            ("t1:", "t1:1", 0.0),  # t1: skipped. Gen keys {}. Ref keys {t1}. I=0, U=1
            ("t1:10, t1:20", "t1:1", 1.0),  # Duplicate gen key. Gen keys {t1}
            # Example from prompt
            (
                "t_1: 70, t_2: 72, t_3: 75",
                "t_1: 70, t_3: 70",
                2 / 3,
            ),  # G={t1,t2,t3}, R={t1,t3}. I=2, U=3
        ],
    )
    def test_single_calculation(self, metric, gen_ts, ref_ts, expected_score):
        score = metric.calculate(gen_ts, ref_ts)
        assert isinstance(score, float)
        assert score == pytest.approx(expected_score)
        assert 0.0 <= score <= 1.0

    def test_batch_calculation_list(self, metric):
        gens = ["t1:1,t2:2", "t1:1,t3:3"]
        refs = ["t1:1,t2:2,t4:4", "t3:3"]
        expected_scores = [
            metric.calculate("t1:1,t2:2", "t1:1,t2:2,t4:4"),  # I=2, U=3 -> 2/3
            metric.calculate("t1:1,t3:3", "t3:3"),  # I=1, U=2 -> 1/2
        ]
        scores = metric.calculate(gens, refs)
        assert isinstance(scores, list)
        assert len(scores) == len(expected_scores)
        for s, e in zip(scores, expected_scores):
            assert s == pytest.approx(e)

    def test_batch_calculation_numpy(self, metric):
        gens_np = np.array(["t1:1,t2:2", "t1:1,t3:3"])
        refs_np = np.array(["t1:1,t2:2,t4:4", "t3:3"])
        expected_scores_np = np.array(
            [
                metric.calculate("t1:1,t2:2", "t1:1,t2:2,t4:4"),
                metric.calculate("t1:1,t3:3", "t3:3"),
            ],
            dtype=float,
        )
        scores = metric.calculate(gens_np, refs_np)
        assert isinstance(scores, np.ndarray)
        np.testing.assert_array_almost_equal(scores, expected_scores_np)

    def test_batch_calculation_pandas(self, metric):
        gens_pd = pd.Series(["t1:1,t2:2", "t1:1,t3:3"])
        refs_pd = pd.Series(["t1:1,t2:2,t4:4", "t3:3"])
        expected_scores_pd = pd.Series(
            [
                metric.calculate("t1:1,t2:2", "t1:1,t2:2,t4:4"),
                metric.calculate("t1:1,t3:3", "t3:3"),
            ],
            index=gens_pd.index,
            dtype=float,
        )
        scores = metric.calculate(gens_pd, refs_pd)
        assert isinstance(scores, pd.Series)
        pd.testing.assert_series_equal(scores, expected_scores_pd, check_dtype=False, atol=1e-6)

    def test_parsing_warnings_skipped_pairs(self, metric):
        gen = "t1:10, t2:bad_val, t3:valid_val_but_not_float"
        ref = "t1:10"

        with pytest.warns(UserWarning) as record:
            score = metric.calculate(gen, ref)

        assert score == pytest.approx(1.0)

        # Check that the correct number of warnings were issued
        assert len(record) == 2
        # Check the content of the warnings (order might vary, so check presence)
        warning_messages = [str(w.message) for w in record]
        assert (
            "Warning: Could not parse value 'bad_val' for key 't2' in time series pair 't2:bad_val'. Skipping."
            in warning_messages
        )
        assert (
            "Warning: Could not parse value 'valid_val_but_not_float' for key 't3' in time series pair 't3:valid_val_but_not_float'. Skipping."
            in warning_messages
        )

        gen2 = "t1:10, :empty_key, key_no_val:, final_key:1.0"
        ref2 = "t1:10, final_key:2.0"

        with pytest.warns(UserWarning) as record2:
            score2 = metric.calculate(gen2, ref2)

        assert score2 == pytest.approx(1.0)
        assert len(record2) == 2
        warning_messages2 = [str(w.message) for w in record2]
        print(warning_messages2)
        assert "Warning: Empty key in time series pair ':empty_key'. Skipping." in warning_messages2
        assert (
            "Warning: Could not parse value '' for key 'key_no_val' in time series pair 'key_no_val:'. Skipping."
            in warning_messages2
        )

    # ** Tests for BaseMetric.calculate behavior (similar to ActionSequenceDiff) **
    def test_calculate_broadcast_gen_str_ref_list(self, metric):
        gen = "t1:1"
        refs = ["t1:1,t2:2", "t3:3"]  # Gen keys {t1}
        expected_scores = [
            metric.calculate("t1:1", "t1:1,t2:2"),  # I=1, U=2 -> 0.5
            metric.calculate("t1:1", "t3:3"),  # I=0, U=2 -> 0.0
        ]
        scores = metric.calculate(gen, refs)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)

    def test_calculate_broadcast_gen_list_ref_str(self, metric):
        gens = ["t1:1,t2:2", "t3:3"]
        ref = "t1:1"  # Ref keys {t1}
        expected_scores = [
            metric.calculate("t1:1,t2:2", "t1:1"),  # I=1, U=2 -> 0.5
            metric.calculate("t3:3", "t1:1"),  # I=0, U=2 -> 0.0
        ]
        scores = metric.calculate(gens, ref)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)

    def test_calculate_missing_reference_uses_first_gen_single_valid(self, metric):
        score = metric.calculate("t1:1,t2:2", None)  # Uses "t1:1,t2:2" as ref
        assert score == pytest.approx(1.0)

        score_empty_ref_str = metric.calculate("t1:1,t2:2", "")  # Uses "t1:1,t2:2" as ref
        assert score_empty_ref_str == pytest.approx(1.0)

    def test_calculate_missing_reference_gen_empty_raises_error(self, metric):
        with pytest.raises(
            ValueError,
            match="generated_texts cannot consist solely of empty or whitespace strings if reference is to be derived",
        ):
            metric.calculate("", None)
        with pytest.raises(
            ValueError,
            match="generated_texts cannot consist solely of empty or whitespace strings if reference is to be derived",
        ):
            metric.calculate("   ", None)
        with pytest.raises(
            ValueError,
            match="generated_texts cannot consist solely of empty or whitespace strings if reference is to be derived",
        ):
            metric.calculate(["", "  "], None)

    def test_calculate_missing_reference_uses_first_gen_batch(self, metric):
        gens = ["t1:1,t2:2", "t3:3"]  # First gen "t1:1,t2:2" becomes ref
        expected_scores = [
            metric.calculate("t1:1,t2:2", "t1:1,t2:2"),  # 1.0
            metric.calculate("t3:3", "t1:1,t2:2"),  # I=0, U=3 -> 0.0
        ]
        scores = metric.calculate(gens, None)
        assert isinstance(scores, list)
        assert scores == pytest.approx(expected_scores)
