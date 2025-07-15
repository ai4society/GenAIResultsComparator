import numpy as np
import pytest
import tempfile
import os
import warnings

from gaico.metrics.audio import AudioSNRNormalized, AudioSpectrogramDistance
from gaico.metrics.audio.audio import __audio_deps_available__

# Skip all tests if audio dependencies are not available
pytestmark = pytest.mark.skipif(
    not __audio_deps_available__,
    reason="Audio dependencies (scipy, soundfile) not installed, skipping audio metrics tests",
)


class TestAudioSNRNormalized:
    @pytest.fixture(scope="class")
    def snr_metric(self):
        return AudioSNRNormalized()

    @pytest.fixture
    def test_signals(self):
        """Generate various test signals."""
        # Pure sine wave
        t = np.linspace(0, 1, 44100)
        clean_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Add different levels of noise
        low_noise = clean_signal + 0.01 * np.random.randn(44100)
        medium_noise = clean_signal + 0.1 * np.random.randn(44100)
        high_noise = clean_signal + 0.5 * np.random.randn(44100)

        return {
            "clean": clean_signal,
            "low_noise": low_noise.astype(np.float32),
            "medium_noise": medium_noise.astype(np.float32),
            "high_noise": high_noise.astype(np.float32),
            "white_noise": np.random.randn(44100).astype(np.float32),
        }

    def test_init_parameters(self):
        # Test custom parameters
        metric = AudioSNRNormalized(snr_min=-30, snr_max=50)
        assert metric.snr_min == -30
        assert metric.snr_max == 50

        # Test invalid parameters
        with pytest.raises(ValueError, match="snr_min .* must be less than snr_max"):
            AudioSNRNormalized(snr_min=50, snr_max=30)

    def test_identical_signals(self, snr_metric, test_signals):
        # Identical signals should have perfect SNR
        signal = test_signals["clean"]
        score = snr_metric.calculate(signal, signal)
        assert score == pytest.approx(1.0)

    def test_noisy_signals(self, snr_metric, test_signals):
        # Test different noise levels
        clean = test_signals["clean"]

        # Low noise should have high score
        score_low = snr_metric.calculate(test_signals["low_noise"], clean)
        assert 0.8 < score_low <= 1.0

        # Medium noise should have medium score
        score_medium = snr_metric.calculate(test_signals["medium_noise"], clean)
        assert 0.4 < score_medium < 0.8

        # High noise should have low score
        score_high = snr_metric.calculate(test_signals["high_noise"], clean)
        assert 0.0 <= score_high < 0.4

    def test_completely_different_signals(self, snr_metric, test_signals):
        clean = test_signals["clean"]
        noise = test_signals["white_noise"]

        score = snr_metric.calculate(noise, clean)
        assert score < 0.3  # Should be very low

    def test_empty_audio_handling(self, snr_metric):
        # Empty numpy array
        with pytest.raises(ValueError, match="Audio array is empty"):
            snr_metric.calculate(np.array([]), np.array([1, 2, 3]))

        # Empty list
        with pytest.raises(
            ValueError,
            match="Batch inputs: generated .* and reference .* must have the same length.",
        ):
            snr_metric.calculate([], [1, 2, 3])

    def test_mismatched_lengths(self, snr_metric):
        # Should handle by truncating to shorter length
        long_signal = np.random.randn(1000)
        short_signal = np.random.randn(500)

        # Should not raise error
        score = snr_metric.calculate(long_signal, short_signal)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_file_input(self, snr_metric, test_signals):
        # Create temporary audio files
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
                try:
                    signal = test_signals["clean"]
                    sf.write(f1.name, signal, 44100)
                    sf.write(f2.name, signal, 44100)

                    score = snr_metric.calculate(f1.name, f2.name)
                    assert score == pytest.approx(1.0)

                finally:
                    os.unlink(f1.name)
                    os.unlink(f2.name)

    def test_file_not_found(self, snr_metric):
        with pytest.raises(FileNotFoundError, match="Audio file not found or is invalid"):
            snr_metric.calculate("nonexistent.wav", np.array([1, 2, 3]))

    def test_invalid_input_type(self, snr_metric):
        with pytest.raises(TypeError, match="Unsupported audio input type"):
            snr_metric.calculate({"not": "audio"}, np.array([1, 2, 3]))

    def test_batch_error_handling(self, snr_metric):
        # Include one invalid input
        gen = [np.array([1, 2, 3]), "invalid", np.array([4, 5, 6])]
        ref = [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([4, 5, 6])]

        with warnings.catch_warnings(record=True) as w:
            scores = snr_metric.calculate(gen, ref)
            assert len(w) == 1
            assert "Error processing audio item" in str(w[0].message)

        assert len(scores) == 3
        assert isinstance(scores[0], float)
        assert scores[1] == 0.0  # Failed item gets 0 score

    def test_resampling_on_init(self):
        """Test that resampling is triggered when sample_rate is set on init."""
        # Metric is set to resample everything to 16000 Hz
        metric = AudioSNRNormalized(sample_rate=16000)

        # Create two identical signals but one will be resampled
        signal1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        signal2 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))

        # The metric should handle the resampling and see them as identical
        score = metric.calculate(signal1, signal2)
        assert score == pytest.approx(1.0)

    def test_batch_with_mixed_types(self, snr_metric, test_signals):
        """Test a batch calculation with mixed input types (file and array)."""
        import soundfile as sf

        clean_signal = test_signals["clean"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            try:
                sf.write(f.name, test_signals["low_noise"], 44100)

                generated = [f.name, test_signals["medium_noise"]]
                references = [clean_signal, clean_signal]

                scores = snr_metric.calculate(generated, references)
                assert isinstance(scores, list)
                assert len(scores) == 2
                assert scores[0] > scores[1]  # low noise should be better than medium
            finally:
                os.unlink(f.name)

    def test_missing_reference_single_audio(self, snr_metric, test_signals):
        """Test that missing reference uses the generated audio itself."""
        signal = test_signals["clean"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score = snr_metric.calculate(signal, reference=None)

            # Should show warning
            assert len(w) == 1
            assert "Reference audio not provided" in str(w[0].message)
            assert "Using the generated audio as reference" in str(w[0].message)

        # Should return 1.0 when comparing with itself
        assert score == 1.0

    def test_missing_reference_batch_list(self, snr_metric, test_signals):
        """Test that missing reference uses first item in batch for lists."""
        batch = [test_signals["clean"], test_signals["low_noise"], test_signals["medium_noise"]]

        # Capture stdout for warning message
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            scores = snr_metric.calculate(batch, reference=None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Should show warning about using first element
        assert "Reference is missing or effectively empty" in output
        assert "Using the first element of `generated` as reference" in output

        # First should be 1.0 (compared with itself)
        assert scores[0] == 1.0

        # Others should be the same as comparing with first
        expected_scores = snr_metric.calculate(batch, batch[0])
        np.testing.assert_array_almost_equal(scores, expected_scores)

    def test_missing_reference_batch_2d_numpy(self, snr_metric, test_signals):
        """Test missing reference with 2D numpy array (batch)."""
        # Create 2D array: each row is an audio signal
        batch_2d = np.array(
            [test_signals["clean"], test_signals["low_noise"], test_signals["medium_noise"]]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = snr_metric.calculate(batch_2d, reference=None)

            # Should show warning
            warning_messages = [str(warning.message) for warning in w]
            assert any("Reference audio not provided" in msg for msg in warning_messages)
            assert any(
                "Using the first generated audio as reference" in msg for msg in warning_messages
            )

        # Should return numpy array
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3

        # First should be 1.0
        assert scores[0] == pytest.approx(1.0)

        # Verify against manual calculation
        expected_scores = snr_metric.calculate(batch_2d, batch_2d[0])
        np.testing.assert_array_almost_equal(scores, expected_scores)

    def test_missing_reference_empty_input(self, snr_metric):
        """Test error handling for empty input with no reference."""
        with pytest.raises(
            ValueError, match="`reference` is missing and cannot be derived from `generated`"
        ):
            snr_metric.calculate([], reference=None)

    def test_missing_reference_pandas_series(self, snr_metric, test_signals):
        """Test missing reference with pandas Series."""
        import pandas as pd

        series_data = pd.Series(
            [test_signals["clean"], test_signals["low_noise"], test_signals["medium_noise"]],
            index=["clean", "low_noise", "medium_noise"],
        )

        # Capture stdout for warning message
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)  # Ignore pandas warnings
                scores = snr_metric.calculate(series_data, reference=None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Should show warning
        assert "Reference is missing or effectively empty" in output

        # Should return pandas Series
        assert isinstance(scores, pd.Series)
        assert len(scores) == 3

        # First should be 1.0
        assert scores.iloc[0] == pytest.approx(1.0)

    def test_missing_reference_single_batch_item(self, snr_metric, test_signals):
        """Test missing reference with single item in batch."""
        single_item_batch = [test_signals["clean"]]

        # Capture stdout for warning message
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore processing warnings
                scores = snr_metric.calculate(single_item_batch, reference=None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Should show warning
        assert "Reference is missing or effectively empty" in output

        # Should return list with single perfect score
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0)


class TestSpectrogramDistance:
    @pytest.fixture(scope="class")
    def spec_metric(self):
        return AudioSpectrogramDistance()

    @pytest.fixture
    def test_signals(self):
        """Generate test signals with different spectral characteristics."""
        t = np.linspace(0, 1, 44100)

        # Different frequencies
        f_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        f_880 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        f_220 = np.sin(2 * np.pi * 220 * t).astype(np.float32)

        # Complex signal with harmonics
        complex_signal = (
            np.sin(2 * np.pi * 440 * t)
            + 0.5 * np.sin(2 * np.pi * 880 * t)
            + 0.25 * np.sin(2 * np.pi * 1320 * t)
        ).astype(np.float32)

        return {
            "f_440": f_440,
            "f_880": f_880,
            "f_220": f_220,
            "complex": complex_signal,
            "white_noise": np.random.randn(44100).astype(np.float32),
        }

    def test_init_parameters(self):
        # Test custom parameters
        metric = AudioSpectrogramDistance(
            n_fft=4096, hop_length=1024, distance_type="cosine", window="hamming"
        )
        assert metric.n_fft == 4096
        assert metric.hop_length == 1024
        assert metric.distance_type == "cosine"
        assert metric.window == "hamming"

        # Test invalid distance type
        with pytest.raises(ValueError, match="Invalid distance_type"):
            AudioSpectrogramDistance(distance_type="invalid")

    def test_identical_signals(self, spec_metric, test_signals):
        signal = test_signals["f_440"]
        score = spec_metric.calculate(signal, signal)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_different_frequencies(self, spec_metric, test_signals):
        # Different frequencies should have lower similarity
        score_octave = spec_metric.calculate(test_signals["f_440"], test_signals["f_880"])
        assert score_octave < 0.8

        score_double_octave = spec_metric.calculate(test_signals["f_220"], test_signals["f_880"])
        assert score_double_octave < score_octave

    def test_distance_types(self, test_signals):
        signal1 = test_signals["f_440"]
        signal2 = test_signals["complex"]

        # Test different distance types
        for distance_type in ["euclidean", "cosine", "correlation"]:
            metric = AudioSpectrogramDistance(distance_type=distance_type)
            score = metric.calculate(signal1, signal2)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_window_types(self, test_signals):
        """Test that different window functions run and produce valid scores."""
        signal1 = test_signals["f_440"]
        signal2 = test_signals["complex"]

        for window in ["hann", "hamming", "blackman"]:
            metric = AudioSpectrogramDistance(window=window)
            score = metric.calculate(signal1, signal2)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_empty_spectrogram_handling(self, spec_metric):
        # Very short signal might produce empty spectrogram
        short_signal = np.array([0.1, 0.2])
        normal_signal = np.random.randn(1000)

        # Should handle gracefully
        with pytest.raises(ValueError):
            spec_metric.calculate(short_signal, normal_signal)

    def test_batch_calculate_list(self, spec_metric, test_signals):
        generated = [test_signals["f_440"], test_signals["f_880"], test_signals["complex"]]
        references = [test_signals["f_440"], test_signals["f_440"], test_signals["f_440"]]

        scores = spec_metric.calculate(generated, references)
        assert isinstance(scores, list)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0, abs=1e-5)  # Identical
        assert scores[1] < scores[0]  # Different frequency

    def test_zero_norm_handling(self, spec_metric):
        # Test with zero signal (should handle division by zero)
        signal_length = 4096  # Must be >= n_fft (default 2048)
        zero_signal = np.zeros(signal_length)
        normal_signal = np.random.randn(signal_length)

        score = spec_metric.calculate(zero_signal, normal_signal)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_correlation_distance_edge_cases(self):
        # Test correlation distance with constant signals
        metric = AudioSpectrogramDistance(distance_type="correlation")

        signal_length = 4096  # Must be >= n_fft (default 2048)
        constant1 = np.ones(signal_length)
        constant2 = np.ones(signal_length) * 2

        # Constant signals have zero variance, should handle gracefully
        score = metric.calculate(constant1, constant2)
        assert isinstance(score, float)

    def test_missing_reference_single_audio(self, spec_metric, test_signals):
        """Test that missing reference uses the generated audio itself."""
        signal = test_signals["f_440"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score = spec_metric.calculate(signal, reference=None)

            # Should show warning
            assert len(w) == 1
            assert "Reference audio not provided" in str(w[0].message)
            assert "Using the generated audio as reference" in str(w[0].message)

        # Should return 1.0 when comparing with itself
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_missing_reference_batch_list(self, spec_metric, test_signals):
        """Test that missing reference uses first item in batch for lists."""
        batch = [test_signals["f_440"], test_signals["f_880"], test_signals["complex"]]

        # Capture stdout for warning message
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            scores = spec_metric.calculate(batch, reference=None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Should show warning about using first element
        assert "Reference is missing or effectively empty" in output
        assert "Using the first element of `generated` as reference" in output

        # First should be 1.0 (compared with itself)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

        # Others should be the same as comparing with first
        expected_scores = spec_metric.calculate(batch, batch[0])
        np.testing.assert_array_almost_equal(scores, expected_scores)

    def test_missing_reference_batch_2d_numpy(self, spec_metric, test_signals):
        """Test missing reference with 2D numpy array (batch)."""
        # Create 2D array: each row is an audio signal
        batch_2d = np.array([test_signals["f_440"], test_signals["f_880"], test_signals["complex"]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = spec_metric.calculate(batch_2d, reference=None)

            # Should show warning
            warning_messages = [str(warning.message) for warning in w]
            assert any("Reference audio not provided" in msg for msg in warning_messages)
            assert any(
                "Using the first generated audio as reference" in msg for msg in warning_messages
            )

        # Should return numpy array
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3

        # First should be 1.0
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

        # Verify against manual calculation
        expected_scores = spec_metric.calculate(batch_2d, batch_2d[0])
        np.testing.assert_array_almost_equal(scores, expected_scores)

    def test_missing_reference_empty_input(self, spec_metric):
        """Test error handling for empty input with no reference."""
        with pytest.raises(
            ValueError, match="`reference` is missing and cannot be derived from `generated`"
        ):
            spec_metric.calculate([], reference=None)

    def test_missing_reference_pandas_series(self, spec_metric, test_signals):
        """Test missing reference with pandas Series."""
        import pandas as pd

        series_data = pd.Series(
            [test_signals["f_440"], test_signals["f_880"], test_signals["complex"]],
            index=["f_440", "f_880", "complex"],
        )

        # Capture stdout for warning message
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)  # Ignore pandas warnings
                scores = spec_metric.calculate(series_data, reference=None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Should show warning
        assert "Reference is missing or effectively empty" in output

        # Should return pandas Series
        assert isinstance(scores, pd.Series)
        assert len(scores) == 3

        # First should be 1.0
        assert scores.iloc[0] == pytest.approx(1.0, abs=1e-5)

    def test_missing_reference_single_batch_item(self, spec_metric, test_signals):
        """Test missing reference with single item in batch."""
        single_item_batch = [test_signals["f_440"]]

        # Capture stdout for warning message
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore processing warnings
                scores = spec_metric.calculate(single_item_batch, reference=None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Should show warning
        assert "Reference is missing or effectively empty" in output

        # Should return list with single perfect score
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0, abs=1e-5)


# Test for missing dependencies
def test_import_error_handling():
    """Test that proper ImportError is raised when dependencies are missing."""
    # This test only runs if dependencies ARE available,
    # so we mock the availability flag
    import gaico.metrics.audio.audio as audio_module

    original_flag = audio_module.__audio_deps_available__
    try:
        audio_module.__audio_deps_available__ = False

        with pytest.raises(ImportError, match="Audio processing dependencies"):
            AudioSNRNormalized()

        with pytest.raises(ImportError, match="Audio processing dependencies"):
            AudioSpectrogramDistance()

    finally:
        audio_module.__audio_deps_available__ = original_flag
