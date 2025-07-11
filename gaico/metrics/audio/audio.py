from abc import ABC
from typing import Any, Iterable, List, Optional
import warnings
import numpy as np
import pandas as pd

from ..base import BaseMetric

# Conditional imports for audio processing
_soundfile = None
_scipy_signal = None
_audio_deps_available = False

try:
    import soundfile as _imported_soundfile
    from scipy import signal as _imported_scipy_signal

    _soundfile = _imported_soundfile
    _scipy_signal = _imported_scipy_signal
    _audio_deps_available = True
except ImportError:
    pass

__audio_deps_available__ = _audio_deps_available


class AudioMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate on audio data.
    Input can be various audio representations (e.g., np.array waveform, audio path).
    """

    def __init__(self, sample_rate: Optional[int] = None, **kwargs: Any):
        """
        Initialize the AudioMetric base class.

        :param sample_rate: Target sample rate for audio processing. If None, uses native sample rate.
        :type sample_rate: Optional[int]
        :param kwargs: Additional keyword arguments
        :type kwargs: Any
        """
        super().__init__()
        self.sample_rate = sample_rate

    def calculate(
        self,
        generated: Any,
        reference: Optional[Any],
        **kwargs: Any,
    ) -> Any:
        """
        Override the base calculate method to handle single audio numpy arrays correctly.
        """
        # For audio, if the input is not a list/tuple/Series, treat it as a single item.
        # This prevents single audio arrays from being iterated over by the base method.
        if not isinstance(generated, (list, tuple, pd.Series)):
            return self._single_calculate(generated, reference, **kwargs)

        # For all other cases (lists of files, etc.), use the default batch logic.
        return super().calculate(generated, reference, **kwargs)

    def _load_audio(self, audio_input: Any) -> tuple[np.ndarray, int]:
        """
        Load audio from various input formats.

        :param audio_input: The audio input (numpy array, file path, or list/tuple of samples)
        :type audio_input: Any
        :return: Tuple of (audio_array, sample_rate)
        :rtype: tuple[np.ndarray, int]
        :raises TypeError: If audio input type is not supported
        :raises ValueError: If audio is empty or invalid
        :raises FileNotFoundError: If audio file path doesn't exist
        """
        if isinstance(audio_input, np.ndarray):
            if audio_input.size == 0:
                raise ValueError("Audio array is empty")
            if audio_input.ndim > 2:
                raise ValueError(
                    f"Audio array has too many dimensions: {audio_input.ndim}. Expected 1D or 2D."
                )
            # If stereo, convert to mono
            if audio_input.ndim == 2:
                audio_input = np.mean(audio_input, axis=0)
            return audio_input.astype(np.float32), self.sample_rate or 44100

        elif isinstance(audio_input, (list, tuple)):
            if len(audio_input) == 0:
                raise ValueError("Audio input list/tuple is empty")
            return np.array(audio_input, dtype=np.float32), self.sample_rate or 44100

        elif isinstance(audio_input, str):
            # Load from file path
            if not _soundfile:
                raise ImportError("soundfile is required to load audio files")
            try:
                audio, sr = _soundfile.read(audio_input, dtype="float32")
                if audio.size == 0:
                    raise ValueError(f"Audio file '{audio_input}' is empty")

                # If stereo, convert to mono
                if audio.ndim == 2:
                    audio = np.mean(audio, axis=1)

                # Resample if needed
                if self.sample_rate and sr != self.sample_rate:
                    audio = self._resample_audio(audio, sr, self.sample_rate)
                    sr = self.sample_rate

                return audio.astype(np.float32), sr

            except Exception as e:
                # Check for soundfile-specific error first
                if isinstance(e, _soundfile.LibsndfileError):
                    if "No such file or directory" in str(e) or "System error" in str(e):
                        raise FileNotFoundError(
                            f"Audio file not found or is invalid: {audio_input}"
                        ) from e
                # Check for generic file not found error
                if isinstance(e, FileNotFoundError):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}") from e

                # Fallback for other errors
                raise ValueError(f"Error loading audio file '{audio_input}': {str(e)}") from e
        else:
            raise TypeError(
                f"Unsupported audio input type: {type(audio_input)}. "
                f"Expected numpy array, list, tuple, or file path string."
            )

    def _ensure_same_length(
        self, audio1: np.ndarray, audio2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensure two audio arrays have the same length by truncating the longer one.

        :param audio1: First audio array
        :type audio1: np.ndarray
        :param audio2: Second audio array
        :type audio2: np.ndarray
        :return: Tuple of audio arrays with same length
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        if len(audio1) == len(audio2):
            return audio1, audio2

        min_len = min(len(audio1), len(audio2))
        if min_len == 0:
            raise ValueError("One or both audio arrays have zero length")

        return audio1[:min_len], audio2[:min_len]

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using scipy if sample rates differ."""
        if orig_sr == target_sr:
            return audio

        if not _scipy_signal:
            raise ImportError("Scipy is required for resampling.")

        num_samples = int(len(audio) * float(target_sr) / orig_sr)
        resampled_audio = _scipy_signal.resample(audio, num_samples)
        return resampled_audio.astype(np.float32)


class AudioSNRNormalized(AudioMetric):
    """
    Normalized Signal-to-Noise Ratio (SNR) metric for audio comparison.

    This metric calculates the SNR between generated and reference audio signals,
    then normalizes it to a 0-1 range for consistency with other GAICo metrics.

    The SNR is calculated as: 20 * log10(RMS_signal / RMS_noise)
    where noise = generated - reference

    Normalization maps SNR values to [0, 1] using linear scaling with clipping.
    Higher scores indicate better quality (less noise).

    Attributes:
        snr_min (float): Minimum expected SNR value for normalization (maps to 0)
        snr_max (float): Maximum expected SNR value for normalization (maps to 1)
        epsilon (float): Small value to prevent division by zero
    """

    def __init__(
        self,
        snr_min: float = -20.0,
        snr_max: float = 40.0,
        epsilon: float = 1e-10,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the AudioSNRNormalized metric.

        :param snr_min: Minimum expected SNR value in dB for normalization (maps to 0).
                        Default -20 dB represents very noisy audio.
        :type snr_min: float
        :param snr_max: Maximum expected SNR value in dB for normalization (maps to 1).
                        Default 40 dB represents very clean audio.
        :type snr_max: float
        :param epsilon: Small value to prevent division by zero in SNR calculation
        :type epsilon: float
        :param sample_rate: Target sample rate for audio processing. If None, uses native sample rate.
        :type sample_rate: Optional[int]
        :param kwargs: Additional parameters passed to parent class
        :type kwargs: Any
        :raises ImportError: If audio processing dependencies are not installed
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if not _audio_deps_available:
            raise ImportError(
                "Audio processing dependencies (scipy, soundfile) are not installed. "
                "Please install them with: pip install 'gaico[audio]'"
            )

        if snr_min >= snr_max:
            raise ValueError(f"snr_min ({snr_min}) must be less than snr_max ({snr_max})")

        self.snr_min = snr_min
        self.snr_max = snr_max
        self.epsilon = epsilon

    def _calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Calculate SNR in dB between signal and noise.

        :param signal: Reference signal array
        :type signal: np.ndarray
        :param noise: Noise array (typically generated - reference)
        :type noise: np.ndarray
        :return: SNR value in decibels
        :rtype: float
        """
        signal_power = np.mean(signal**2) + self.epsilon
        noise_power = np.mean(noise**2) + self.epsilon

        # If noise is essentially zero, return max SNR
        if noise_power <= self.epsilon:
            return self.snr_max

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    def _normalize_snr(self, snr_db: float) -> float:
        """
        Normalize SNR from dB to [0, 1] range.

        :param snr_db: SNR value in decibels
        :type snr_db: float
        :return: Normalized SNR score between 0 and 1
        :rtype: float
        """
        # Linear normalization with clipping
        normalized = (snr_db - self.snr_min) / (self.snr_max - self.snr_min)
        return float(np.clip(normalized, 0.0, 1.0))

    def _single_calculate(self, generated_item: Any, reference_item: Any, **kwargs: Any) -> float:
        """
        Calculate normalized SNR for a single pair of audio signals.

        :param generated_item: The generated audio (numpy array, file path, or list)
        :type generated_item: Any
        :param reference_item: The reference audio (numpy array, file path, or list)
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (not used)
        :type kwargs: Any
        :return: Normalized SNR score between 0 and 1
        :rtype: float
        :raises ValueError: If audio cannot be loaded or processed
        :raises TypeError: If audio input type is not supported
        """
        try:
            # Load audio
            gen_audio, gen_sr = self._load_audio(generated_item)
            ref_audio, ref_sr = self._load_audio(reference_item)

            # Resample if necessary
            if self.sample_rate and gen_sr != self.sample_rate:
                gen_audio = self._resample_audio(gen_audio, gen_sr, self.sample_rate)
                gen_sr = self.sample_rate
            if self.sample_rate and ref_sr != self.sample_rate:
                ref_audio = self._resample_audio(ref_audio, ref_sr, self.sample_rate)
                ref_sr = self.sample_rate

            if gen_sr != ref_sr:
                warnings.warn(
                    f"Sample rates differ (generated: {gen_sr}, reference: {ref_sr}). "
                    f"Resampling generated audio to match reference rate {ref_sr} Hz."
                )
                gen_audio = self._resample_audio(gen_audio, orig_sr=gen_sr, target_sr=ref_sr)

            # Ensure same length
            gen_audio, ref_audio = self._ensure_same_length(gen_audio, ref_audio)

            # Calculate noise as difference
            noise = gen_audio - ref_audio

            # Calculate SNR
            snr_db = self._calculate_snr(ref_audio, noise)

            # Normalize to [0, 1]
            return self._normalize_snr(snr_db)

        except (ValueError, TypeError):
            raise
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ValueError(f"Error calculating audio SNR: {str(e)}")

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | np.ndarray | pd.Series:
        """
        Calculate normalized SNR for a batch of audio signals.

        :param generated_items: Iterable of generated audio items
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference audio items
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments passed to _single_calculate
        :type kwargs: Any
        :return: List, array, or Series of normalized SNR scores
        :rtype: List[float] | np.ndarray | pd.Series
        :raises ValueError: If input lengths don't match
        """
        results = []
        for gen, ref in zip(generated_items, reference_items):
            try:
                score = self._single_calculate(gen, ref, **kwargs)
                results.append(score)
            except Exception as e:
                import warnings

                warnings.warn(f"Error processing audio item: {str(e)}. Setting score to 0.0")
                results.append(0.0)

        if isinstance(generated_items, np.ndarray):
            return np.array(results, dtype=float)
        if isinstance(generated_items, pd.Series):
            return pd.Series(results, index=generated_items.index, dtype=float)
        return results


class AudioSpectrogramDistance(AudioMetric):
    """
    Spectrogram-based distance metric for audio comparison.

    This metric computes spectrograms of audio signals and calculates
    the distance between them using various distance measures. The distance
    is then converted to a similarity score in the range [0, 1].

    Spectrograms capture the frequency content of audio over time, making
    this metric suitable for comparing timbral and spectral characteristics.

    Attributes:
        n_fft (int): FFT window size for spectrogram computation
        hop_length (int): Number of samples between successive frames
        distance_type (str): Type of distance metric to use
        window (str): Window function for STFT
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        distance_type: str = "euclidean",
        window: str = "hann",
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the SpectrogramDistance metric.

        :param n_fft: FFT window size for spectrogram. Larger values give better
                    frequency resolution but worse time resolution.
        :type n_fft: int
        :param hop_length: Number of samples between successive frames. Smaller
                        values give better time resolution.
        :type hop_length: int
        :param distance_type: Type of distance metric. Options: 'euclidean', 'cosine', 'correlation'
        :type distance_type: str
        :param window: Window function for STFT. Options include 'hann', 'hamming', 'blackman'
        :type window: str
        :param sample_rate: Target sample rate for audio processing
        :type sample_rate: Optional[int]
        :param kwargs: Additional parameters passed to parent class
        :type kwargs: Any
        :raises ImportError: If audio processing dependencies are not installed
        :raises ValueError: If distance_type is not supported
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if not _audio_deps_available:
            raise ImportError(
                "Audio processing dependencies (scipy) are not installed. "
                "Please install them with: pip install 'gaico[audio]'"
            )

        valid_distances = ["euclidean", "cosine", "correlation"]
        if distance_type not in valid_distances:
            raise ValueError(
                f"Invalid distance_type '{distance_type}'. Must be one of: {valid_distances}"
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.distance_type = distance_type
        self.window = window

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute magnitude spectrogram of audio signal.

        :param audio: Audio signal array
        :type audio: np.ndarray
        :return: Magnitude spectrogram (frequency x time)
        :rtype: np.ndarray
        :raises ValueError: If spectrogram computation fails
        """
        try:
            if not _scipy_signal:
                raise ImportError("Scipy is required for STFT computation.")

            nperseg = min(self.n_fft, len(audio))
            noverlap = min(self.n_fft - self.hop_length, nperseg - 1)

            f, t, Zxx = _scipy_signal.stft(
                audio,
                fs=self.sample_rate or 44100,
                window=self.window,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            return np.abs(Zxx)
        except Exception as e:
            raise ValueError(f"Failed to compute spectrogram: {str(e)}")

    def _calculate_distance(self, spec1: np.ndarray, spec2: np.ndarray) -> float:
        """
        Calculate distance between two spectrograms.

        :param spec1: First spectrogram
        :type spec1: np.ndarray
        :param spec2: Second spectrogram
        :type spec2: np.ndarray
        :return: Distance value (lower means more similar)
        :rtype: float
        """
        # Ensure same shape
        min_freq_bins = min(spec1.shape[0], spec2.shape[0])
        min_frames = min(spec1.shape[1], spec2.shape[1])
        if min_frames == 0 or min_freq_bins == 0:
            raise ValueError("One or both spectrograms have zero frames or frequency bins")

        spec1 = spec1[:min_freq_bins, :min_frames]
        spec2 = spec2[:min_freq_bins, :min_frames]

        if self.distance_type == "euclidean":
            distance = np.sqrt(np.mean((spec1 - spec2) ** 2))
            # Normalize by average magnitude to avoid scale issues
            norm_factor = (np.mean(spec1) + np.mean(spec2)) / 2 + 1e-10
            return distance / norm_factor

        elif self.distance_type == "cosine":
            # Flatten and compute cosine distance
            spec1_flat = spec1.flatten()
            spec2_flat = spec2.flatten()

            norm1 = np.linalg.norm(spec1_flat)
            norm2 = np.linalg.norm(spec2_flat)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            cosine_sim = np.dot(spec1_flat, spec2_flat) / (norm1 * norm2)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
            return 1.0 - cosine_sim

        elif self.distance_type == "correlation":
            spec1_flat = spec1.flatten()
            spec2_flat = spec2.flatten()

            spec1_centered = spec1_flat - np.mean(spec1_flat)
            spec2_centered = spec2_flat - np.mean(spec2_flat)

            norm1 = np.linalg.norm(spec1_centered)
            norm2 = np.linalg.norm(spec2_centered)

            if norm1 == 0 or norm2 == 0:
                return 0.0 if norm1 == norm2 else 1.0

            correlation = np.dot(spec1_centered, spec2_centered) / (norm1 * norm2)
            correlation = np.clip(correlation, -1.0, 1.0)
            return 1.0 - correlation
        return 0.0  # Default case, should not happen

    def _single_calculate(self, generated_item: Any, reference_item: Any, **kwargs: Any) -> float:
        """
        Calculate spectrogram distance for a single pair of audio signals.

        :param generated_item: The generated audio
        :type generated_item: Any
        :param reference_item: The reference audio
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (not used)
        :type kwargs: Any
        :return: Similarity score between 0 and 1 (1 = identical)
        :rtype: float
        :raises ValueError: If audio cannot be loaded or processed
        :raises TypeError: If audio input type is not supported
        """
        try:
            # Load audio
            gen_audio, gen_sr = self._load_audio(generated_item)
            ref_audio, ref_sr = self._load_audio(reference_item)

            # Ensure audio is long enough for STFT
            if len(gen_audio) < self.n_fft or len(ref_audio) < self.n_fft:
                raise ValueError(
                    f"Audio signal is too short for the given n_fft ({self.n_fft}). "
                    f"Signal length must be >= n_fft."
                )

            # Resample if necessary
            if self.sample_rate and gen_sr != self.sample_rate:
                gen_audio = self._resample_audio(gen_audio, gen_sr, self.sample_rate)
                gen_sr = self.sample_rate
            if self.sample_rate and ref_sr != self.sample_rate:
                ref_audio = self._resample_audio(ref_audio, ref_sr, self.sample_rate)
                ref_sr = self.sample_rate

            if gen_sr != ref_sr:
                warnings.warn(
                    f"Sample rates differ (generated: {gen_sr}, reference: {ref_sr}). "
                    f"Resampling generated audio to match reference rate {ref_sr} Hz."
                )
                gen_audio = self._resample_audio(gen_audio, orig_sr=gen_sr, target_sr=ref_sr)

            # Compute spectrograms
            gen_spec = self._compute_spectrogram(gen_audio)
            ref_spec = self._compute_spectrogram(ref_audio)

            # Calculate distance
            distance = self._calculate_distance(gen_spec, ref_spec)

            # Convert distance to similarity using exponential decay
            # This maps distance [0, inf) to similarity (1, 0]
            similarity = np.exp(-distance)
            return float(np.clip(similarity, 0.0, 1.0))

        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise ValueError(f"Error calculating spectrogram distance: {str(e)}")

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | np.ndarray | pd.Series:
        """
        Calculate spectrogram distance for a batch of audio signals.

        :param generated_items: Iterable of generated audio items
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference audio items
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments passed to _single_calculate
        :type kwargs: Any
        :return: List, array, or Series of similarity scores
        :rtype: List[float] | np.ndarray | pd.Series
        :raises ValueError: If input lengths don't match
        """

        results = []
        for gen, ref in zip(generated_items, reference_items):
            try:
                score = self._single_calculate(gen, ref, **kwargs)
                results.append(score)
            except Exception as e:
                import warnings

                warnings.warn(f"Error processing spectrogram item: {str(e)}. Setting score to 0.0")
                results.append(0.0)

        if isinstance(generated_items, np.ndarray):
            return np.array(results, dtype=float)
        if isinstance(generated_items, pd.Series):
            return pd.Series(results, index=generated_items.index, dtype=float)
        return results
