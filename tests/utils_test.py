from collections import Counter

import numpy as np
import pandas as pd
import pytest

from gaico.utils import (
    batch_get_ngrams,
    generate_deltas_frame,
    get_ngrams,
    prepare_results_dataframe,
    to_iterable,
)


#  Tests for to_iterable
def test_to_iterable():
    assert isinstance(to_iterable(np.array([1, 2])), np.ndarray)
    assert isinstance(to_iterable(pd.Series([1, 2])), pd.Series)
    assert isinstance(to_iterable([1, 2]), list)
    assert isinstance(to_iterable((1, 2)), list)
    assert to_iterable("abc") == ["abc"]  # type: ignore
    assert to_iterable(123) == [123]  # type: ignore
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert isinstance(to_iterable(df), np.ndarray)  # .values
    assert to_iterable({"x": 1, "y": 2}) == [1, 2]  # type: ignore


#  Tests for get_ngrams
def test_get_ngrams():
    text = "This is a test sentence."
    assert get_ngrams(text, 1) == Counter({"this": 1, "is": 1, "a": 1, "test": 1, "sentence.": 1})
    assert get_ngrams(text, 2) == Counter(
        {"this is": 1, "is a": 1, "a test": 1, "test sentence.": 1}
    )
    assert get_ngrams("", 2) == Counter()


#  Tests for batch_get_ngrams
def test_batch_get_ngrams_list():
    texts = ["hello world", "test sentence"]
    result = batch_get_ngrams(texts, 2)
    assert len(result) == 2
    assert result[0] == Counter({"hello world": 1})
    assert result[1] == Counter({"test sentence": 1})


def test_batch_get_ngrams_numpy():
    texts = np.array(["hello world", "test sentence"])
    result = batch_get_ngrams(texts, 1)
    assert len(result) == 2
    assert result[0]["hello"] == 1


def test_batch_get_ngrams_pandas_series():
    texts = pd.Series(["hello world", "test sentence"])
    result = batch_get_ngrams(texts, 1)
    assert len(result) == 2
    assert result[0]["world"] == 1


#  Tests for prepare_results_dataframe
def test_prepare_results_dataframe_basic():
    results_dict = {
        "ModelA": {"BLEU": 0.8, "ROUGE": {"rouge1": 0.75, "rougeL": 0.7}},  # ROUGE has sub-metrics
        "ModelB": {"BLEU": 0.7, "Jaccard": 0.6},  # Jaccard is direct
    }
    df = prepare_results_dataframe(results_dict)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["model_name", "metric_name", "score"]
    assert (
        len(df) == 5
    )  # ModelA-BLEU, ModelA-ROUGE_rouge1, ModelA-ROUGE_rougeL, ModelB-BLEU, ModelB-Jaccard

    model_a_bleu = df[(df["model_name"] == "ModelA") & (df["metric_name"] == "BLEU")]
    assert model_a_bleu["score"].iloc[0] == 0.8

    model_a_rouge1 = df[(df["model_name"] == "ModelA") & (df["metric_name"] == "ROUGE_rouge1")]
    assert model_a_rouge1["score"].iloc[0] == 0.75

    model_b_jaccard = df[(df["model_name"] == "ModelB") & (df["metric_name"] == "Jaccard")]
    assert model_b_jaccard["score"].iloc[0] == 0.6


def test_prepare_results_dataframe_empty_input():
    df = prepare_results_dataframe({})
    assert df.empty
    assert list(df.columns) == ["model_name", "metric_name", "score"]


def test_prepare_results_dataframe_custom_colnames():
    results_dict = {"M1": {"MetricX": 0.5}}
    df = prepare_results_dataframe(results_dict, model_col="mod", metric_col="met", score_col="val")
    assert list(df.columns) == ["mod", "met", "val"]
    assert df["mod"].iloc[0] == "M1"


def test_prepare_results_dataframe_non_numeric_score_ignored():
    results_dict = {"ModelA": {"BLEU": "high"}}  # Non-numeric score
    df = prepare_results_dataframe(results_dict)
    assert df.empty  # Should ignore non-numeric scores

    results_dict_nested_non_numeric = {"ModelA": {"ROUGE": {"f1": "good"}}}
    df_nested = prepare_results_dataframe(results_dict_nested_non_numeric)
    assert df_nested.empty


#  Tests for generate_deltas_frame
@pytest.fixture
def sample_threshold_results_single():
    return {
        "BLEU": {"score": 0.6, "threshold_applied": 0.5, "passed_threshold": True},
        "ROUGE": {"score": 0.4, "threshold_applied": 0.5, "passed_threshold": False},
    }


@pytest.fixture
def sample_threshold_results_batch():
    return [
        {
            "BLEU": {"score": 0.6, "passed_threshold": True},
            "ROUGE": {"score": 0.4, "passed_threshold": False},
        },
        {
            "BLEU": {"score": 0.3, "passed_threshold": False},
            "ROUGE": {"score": 0.8, "passed_threshold": True},
        },
    ]


def test_generate_deltas_frame_single(sample_threshold_results_single):
    df = generate_deltas_frame(
        sample_threshold_results_single, generated_texts="gen text", reference_texts="ref text"
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "generated_text" in df.columns
    assert "reference_text" in df.columns
    assert "BLEU_score" in df.columns
    assert "BLEU_passed" in df.columns
    assert "ROUGE_score" in df.columns
    assert "ROUGE_passed" in df.columns
    assert df["generated_text"].iloc[0] == "gen text"
    assert df["BLEU_score"].iloc[0] == 0.6
    assert df["ROUGE_passed"].iloc[0] is np.False_


def test_generate_deltas_frame_batch(sample_threshold_results_batch):
    df = generate_deltas_frame(
        sample_threshold_results_batch,
        generated_texts=["gen1", "gen2"],
        reference_texts=["ref1", "ref2"],
    )
    assert len(df) == 2
    assert df["generated_text"].iloc[1] == "gen2"
    assert df["BLEU_score"].iloc[0] == 0.6
    assert df["ROUGE_passed"].iloc[1] is np.True_


def test_generate_deltas_frame_no_texts(sample_threshold_results_single):
    df = generate_deltas_frame(sample_threshold_results_single)
    assert "generated_text" not in df.columns
    assert "reference_text" not in df.columns


def test_generate_deltas_frame_to_csv(tmp_path, sample_threshold_results_single):
    """Test CSV output."""
    csv_file = tmp_path / "report.csv"
    df = generate_deltas_frame(sample_threshold_results_single, output_csv_path=str(csv_file))
    assert csv_file.is_file()
    df_read = pd.read_csv(csv_file)
    assert len(df_read) == len(df)
    assert "BLEU_score" in df_read.columns


def test_generate_deltas_frame_empty_results():
    df = generate_deltas_frame([])
    assert df.empty
    # Check for default columns if no data
    assert "generated_text" in df.columns
    assert "metric_name" in df.columns

    df_dict = generate_deltas_frame({})  # This might be an edge case depending on how it's handled
    assert df_dict.empty
