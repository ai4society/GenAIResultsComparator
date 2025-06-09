from typing import Dict

import pytest

from gaico.thresholds import (
    DEFAULT_THRESHOLD,
    apply_thresholds,
    calculate_pass_fail_percent,
    get_default_thresholds,
)


#  Tests for get_default_thresholds
def test_get_default_thresholds_returns_copy():
    """Test that get_default_thresholds returns a copy, not the original."""
    defaults = get_default_thresholds()
    assert defaults == DEFAULT_THRESHOLD
    defaults["NEW_METRIC"] = 0.99
    assert "NEW_METRIC" not in DEFAULT_THRESHOLD, "get_default_thresholds should return a copy"


#  Tests for apply_thresholds
def test_apply_thresholds_single_dict_default_thresholds():
    """Test apply_thresholds with a single result dictionary and default thresholds."""
    results = {"BLEU": 0.6, "JSD": 0.1, "ROUGE": 0.4}  # JSD is 1-score
    expected_thresholds = get_default_thresholds()
    thresholded = apply_thresholds(results)

    assert isinstance(thresholded, dict)
    assert thresholded["BLEU"]["score"] == 0.6
    assert thresholded["BLEU"]["threshold_applied"] == expected_thresholds["BLEU"]
    assert thresholded["BLEU"]["passed_threshold"] is True

    assert thresholded["JSD"]["score"] == 0.1
    assert thresholded["JSD"]["threshold_applied"] == expected_thresholds["JSD"]
    assert thresholded["JSD"]["passed_threshold"] is True  # (1 - 0.1) = 0.9 >= 0.5

    assert thresholded["ROUGE"]["score"] == 0.4
    assert thresholded["ROUGE"]["threshold_applied"] == expected_thresholds["ROUGE"]
    assert thresholded["ROUGE"]["passed_threshold"] is False


def test_apply_thresholds_single_dict_custom_thresholds():
    """Test apply_thresholds with custom thresholds."""
    results = {"BLEU": 0.6, "JSD": 0.7}
    custom_thresholds = {"BLEU": 0.7, "JSD": 0.2}  # JSD: (1-score) >= threshold
    thresholded: Dict = apply_thresholds(results, thresholds=custom_thresholds)  # type: ignore

    assert thresholded["BLEU"]["passed_threshold"] is False
    assert thresholded["BLEU"]["threshold_applied"] == 0.7
    assert thresholded["JSD"]["passed_threshold"] is True  # (1 - 0.7) = 0.3 >= 0.2
    assert thresholded["JSD"]["threshold_applied"] == 0.2


def test_apply_thresholds_list_of_dicts():
    """Test apply_thresholds with a list of result dictionaries."""
    results_list = [
        {"BLEU": 0.6, "ROUGE": 0.3},
        {"BLEU": 0.4, "ROUGE": 0.7},
    ]
    custom_thresholds = {"BLEU": 0.5, "ROUGE": 0.5}
    thresholded_list = apply_thresholds(results_list, thresholds=custom_thresholds)

    assert isinstance(thresholded_list, list)
    assert len(thresholded_list) == 2

    assert thresholded_list[0]["BLEU"]["passed_threshold"] is True
    assert thresholded_list[0]["ROUGE"]["passed_threshold"] is False
    assert thresholded_list[1]["BLEU"]["passed_threshold"] is False
    assert thresholded_list[1]["ROUGE"]["passed_threshold"] is True


def test_apply_thresholds_metric_not_in_thresholds():
    """Test behavior when a metric in results doesn't have a defined threshold."""
    results = {"BLEU": 0.6, "UNKNOWN_METRIC": 0.9}
    thresholded = apply_thresholds(results)
    assert "BLEU" in thresholded
    assert "UNKNOWN_METRIC" not in thresholded  # Only applies if threshold exists


def test_apply_thresholds_invalid_input_type():
    """Test apply_thresholds with invalid input type."""
    with pytest.raises(TypeError):
        apply_thresholds("not a dict or list")  # type: ignore


#  Tests for calculate_pass_fail_percent
def test_calculate_pass_fail_percent_basic():
    results = {
        "BLEU": [0.6, 0.4, 0.7],  # Default threshold 0.5 -> Pass, Fail, Pass
        "JSD": [
            0.1,
            0.8,
            0.3,
        ],  # Default threshold 0.5 -> (1-0.1)=0.9 P, (1-0.8)=0.2 F, (1-0.3)=0.7 P
    }
    stats = calculate_pass_fail_percent(results)

    assert stats["BLEU"]["total_passed"] == 2
    assert stats["BLEU"]["total_failed"] == 1
    assert stats["BLEU"]["pass_percentage"] == (2 / 3 * 100)

    assert stats["JSD"]["total_passed"] == 2
    assert stats["JSD"]["total_failed"] == 1
    assert stats["JSD"]["pass_percentage"] == (2 / 3 * 100)


def test_calculate_pass_fail_percent_custom_thresholds():
    results = {"ROUGE": [0.6, 0.2]}
    custom_thresholds = {"ROUGE": 0.7}  # Fail, Fail
    stats = calculate_pass_fail_percent(results, thresholds=custom_thresholds)
    assert stats["ROUGE"]["total_passed"] == 0
    assert stats["ROUGE"]["total_failed"] == 2


def test_calculate_pass_fail_percent_empty_results():
    assert calculate_pass_fail_percent({}) == {}
    assert calculate_pass_fail_percent({"BLEU": []}) == {
        "BLEU": {
            "total_passed": 0,
            "total_failed": 0,
            "pass_percentage": 0,
            "fail_percentage": 0,
        }
    }


def test_calculate_pass_fail_percent_metric_no_threshold():
    results = {"BLEU": [0.6], "UNKNOWN": [0.9]}
    stats = calculate_pass_fail_percent(results)
    assert "BLEU" in stats
    assert "UNKNOWN" not in stats
