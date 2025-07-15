from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional
import warnings
import os

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

    This class provides a common framework for audio metrics, including
    handling for various input types (file paths, numpy arrays), audio loading,
    resampling, and ensuring consistent signal lengths for comparison.
    """

    def __init__(self, sample_rate: Optional[int] = None, **kwargs: Any):
        """
        Initialize the AudioMetric base class.

        Parameters
        ----------
        sample_rate : int, optional
            Target sample rate for audio processing. If provided, all audio will be
            resampled to this rate. If None, the native sample rate of the
            reference audio is used. Defaults to None.
        kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        ImportError
            If required audio dependencies (scipy, soundfile) are not installed.
        """
        if not __audio_deps_available__:
            raise ImportError(
                "Audio processing dependencies (scipy, soundfile) not installed. "
                "Please install them with: pip install scipy soundfile"
            )
        super().__init__()
        self.sample_rate = sample_rate

    def calculate(
        self,
        generated: Any,
        reference: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Calculate the audio metric for given generated and reference inputs.

        This method overrides the base `calculate` to intelligently handle
        audio-specific data types like numpy arrays, which the base class
        might misinterpret. It routes single items, batches of arrays, and
        other formats to the appropriate internal calculation methods.

        Parameters
        ----------
        generated : Any
            The generated audio to evaluate. Can be a file path (str), a 1D numpy
            array (single signal), a 2D numpy array (batch of signals), or a
            list/Series of paths or arrays.
        reference : Any, optional
            The reference audio to compare against. Format requirements are the
            same as for `generated`. If None, the generated audio is used as its
            own reference, typically resulting in a perfect score.
        kwargs : Any
            Additional keyword arguments passed to the specific metric's
            calculation logic.

        Returns
        -------
        Any
            The calculated score. This is typically a float for single inputs or a
            list/array/Series of floats for batch inputs.
        """

        # Helper to determine if an input is a valid audio type (path, array, list, or series).
        def is_valid_audio_type(item: Any) -> bool:
            return isinstance(item, (str, np.ndarray, list, pd.Series, tuple))

        # Check if the input type is fundamentally unsupported.
        # This catches invalid types like dictionaries before any other logic.
        if not is_valid_audio_type(generated):
            # This will call _load_audio with the invalid type, which correctly raises TypeError.
            return self._single_calculate(generated, reference, **kwargs)

        # Helper to determine if an input is a single audio entity (path or 1D array).
        def is_single_audio_item(item: Any) -> bool:
            return isinstance(item, str) or (isinstance(item, np.ndarray) and item.ndim == 1)

        # Case 1: The `generated` input is clearly a single audio item.
        if is_single_audio_item(generated):
            actual_ref = reference
            if reference is None:
                warnings.warn(
                    "Reference audio not provided. Using the generated audio as reference.",
                    UserWarning,
                )
                actual_ref = generated
            # Directly call _single_calculate to ensure exceptions are raised correctly.
            return self._single_calculate(generated, actual_ref, **kwargs)

        # Case 2: `generated` is a 2D numpy array, which is always a batch.
        if isinstance(generated, np.ndarray) and generated.ndim == 2:
            ref_list = reference
            if ref_list is None:
                warnings.warn(
                    "Reference audio not provided. Using the first generated audio as reference.",
                    UserWarning,
                )
                ref_list = [generated[0]] * len(generated)
            elif isinstance(ref_list, np.ndarray) and ref_list.ndim == 1:
                ref_list = [ref_list] * len(generated)
            return self._batch_calculate(generated, ref_list, **kwargs)

        # Case 3: `generated` is a list/series containing numpy arrays. BaseMetric will fail. Handle it here.
        is_batch_of_numpy = False
        if isinstance(generated, (list, pd.Series)) and len(generated) > 0:
            first_item = generated[0] if isinstance(generated, list) else generated.iloc[0]
            if isinstance(first_item, np.ndarray):
                is_batch_of_numpy = True

        if is_batch_of_numpy:
            ref_list = reference
            if ref_list is None:
                print(
                    "Warning: Reference is missing or effectively empty. Using the first element of `generated` as reference."
                )
                first_item = generated[0] if isinstance(generated, list) else generated.iloc[0]
                ref_list = [first_item] * len(generated)
            elif isinstance(ref_list, np.ndarray) and ref_list.ndim == 1:
                ref_list = [ref_list] * len(generated)

            return self._batch_calculate(generated, ref_list, **kwargs)

        # For all other cases (e.g., list of file paths), the parent logic is correct.
        return super().calculate(generated, reference, **kwargs)

    @abstractmethod
    def _single_calculate(self, generated_item: Any, reference_item: Any, **kwargs: Any) -> float:
        """
        (Internal) Calculate the metric for a single pair of items.

        This method must be implemented by all concrete subclasses. It defines
        the core logic for comparing one generated audio item to one reference
        audio item. It should raise exceptions on failure (e.g.,
        FileNotFoundError, TypeError) rather than handling them.

        Parameters
        ----------
        generated_item : Any
            A single generated audio item (e.g., file path or numpy array).
        reference_item : Any
            A single reference audio item.
        kwargs : Any
            Metric-specific keyword arguments.

        Returns
        -------
        float
            The calculated score for the single pair.
        """
        pass

    def _batch_calculate(
        self,
        generated_items: Iterable,
        reference_items: Iterable,
        **kwargs: Any,
    ) -> List[float] | np.ndarray | pd.Series:
        """
        (Internal) Calculate the metric for a batch of items.

        This default implementation iterates over the generated and reference
        items, calling `_single_calculate` for each pair. It handles errors
        gracefully by issuing a warning and assigning a score of 0.0 for any
        pair that fails, allowing the batch processing to continue.

        Parameters
        ----------
        generated_items : Iterable
            An iterable of generated audio items.
        reference_items : Iterable
            An iterable of reference audio items.
        kwargs : Any
            Metric-specific keyword arguments.

        Returns
        -------
        List[float] | np.ndarray | pd.Series
            A collection of scores, with the type matching the input `generated_items`
            (defaulting to list).
        """
        results = []
        for gen, ref in zip(generated_items, reference_items):
            try:
                score = self._single_calculate(gen, ref, **kwargs)
                results.append(score)
            except Exception as e:
                warnings.warn(f"Error processing audio item: {str(e)}. Setting score to 0.0")
                results.append(0.0)

        if isinstance(generated_items, np.ndarray):
            return np.array(results, dtype=float)
        if isinstance(generated_items, pd.Series):
            return pd.Series(results, index=generated_items.index, dtype=float)
        return results

    def _load_audio(self, audio_input: Any) -> tuple[np.ndarray, int]:
        """
        Load audio from various input formats into a mono numpy array.

        Parameters
        ----------
        audio_input : Any
            The audio input to load. Supported types are:
            - str: Path to an audio file.
            - np.ndarray: A 1D or 2D numpy array. 2D is averaged to mono.
            - list, tuple: A sequence of numbers representing a waveform.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing:
            - np.ndarray: The loaded audio signal as a 1D float32 numpy array.
            - int: The sample rate of the loaded audio.

        Raises
        ------
        ImportError
            If `soundfile` is required but not installed.
        FileNotFoundError
            If `audio_input` is a path that does not exist.
        ValueError
            If the audio file or array is empty or malformed.
        TypeError
            If `audio_input` is of an unsupported type.
        """
        if isinstance(audio_input, str):
            if not _soundfile:
                raise ImportError("soundfile is required to load audio files")
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found or is invalid: {audio_input}")
            try:
                audio, sr = _soundfile.read(audio_input, dtype="float32")
                if audio.size == 0:
                    raise ValueError(f"Audio file '{audio_input}' is empty")
            except Exception as e:
                raise ValueError(f"Error loading audio file '{audio_input}': {str(e)}") from e

            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            if self.sample_rate and sr != self.sample_rate:
                audio = self._resample_audio(audio, sr, self.sample_rate)
                sr = self.sample_rate
            return audio.astype(np.float32), sr

        if isinstance(audio_input, np.ndarray):
            if audio_input.size == 0:
                raise ValueError("Audio array is empty")
            if audio_input.ndim > 2:
                raise ValueError(
                    f"Audio array has too many dimensions: {audio_input.ndim}. Expected 1D or 2D."
                )
            if audio_input.ndim == 2:
                audio_input = np.mean(audio_input, axis=0)
            return audio_input.astype(np.float32), self.sample_rate or 44100

        if isinstance(audio_input, (list, tuple)):
            if len(audio_input) == 0:
                raise ValueError("Audio input list/tuple is empty")
            return np.array(audio_input, dtype=np.float32), self.sample_rate or 44100

        raise TypeError(
            f"Unsupported audio input type: {type(audio_input)}. Expected numpy array, list, tuple, or file path string."
        )

    def _ensure_same_length(
        self, audio1: np.ndarray, audio2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensure two audio arrays have the same length by truncating the longer one.

        Parameters
        ----------
        audio1 : np.ndarray
            The first audio signal.
        audio2 : np.ndarray
            The second audio signal.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the two audio signals, truncated to the same length.

        Raises
        ------
        ValueError
            If either audio signal has zero length after processing.
        """
        if len(audio1) == len(audio2):
            return audio1, audio2
        min_len = min(len(audio1), len(audio2))
        if min_len == 0:
            raise ValueError("One or both audio arrays have zero length after processing")
        return audio1[:min_len], audio2[:min_len]

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to a target sample rate using scipy.

        Parameters
        ----------
        audio : np.ndarray
            The input audio signal.
        orig_sr : int
            The original sample rate of the audio.
        target_sr : int
            The target sample rate to resample to.

        Returns
        -------
        np.ndarray
            The resampled audio signal.

        Raises
        ------
        ImportError
            If `scipy` is required for resampling but is not installed.
        """
        if orig_sr == target_sr:
            return audio
        if not _scipy_signal:
            raise ImportError("Scipy is required for resampling.")
        num_samples = int(len(audio) * float(target_sr) / orig_sr)
        return _scipy_signal.resample(audio, num_samples).astype(np.float32)


class AudioSNRNormalized(AudioMetric):
    """
    Calculates a normalized Signal-to-Noise Ratio (SNR) between two audio signals.

    The metric computes the standard SNR in decibels (dB) and then scales this
    value to a normalized range of [0, 1]. A score of 1.0 indicates identical
    signals (infinite SNR), while a score of 0.0 indicates high levels of noise
    or distortion, at or below the configured `snr_min` threshold.
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

        Parameters
        ----------
        snr_min : float, optional
            The minimum SNR (in dB) that maps to a normalized score of 0.0.
            Defaults to -20.0.
        snr_max : float, optional
            The maximum SNR (in dB) that maps to a normalized score of 1.0.
            Defaults to 40.0.
        epsilon : float, optional
            A small value added to power calculations to prevent division by zero.
            Defaults to 1e-10.
        sample_rate : int, optional
            Target sample rate for audio processing. Defaults to None.
        kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If `snr_min` is not less than `snr_max`.
        """

        super().__init__(sample_rate=sample_rate, **kwargs)
        if snr_min >= snr_max:
            raise ValueError(f"snr_min ({snr_min}) must be less than snr_max ({snr_max})")
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.epsilon = epsilon

    def get_name(self) -> str:
        return "AudioSNR"

    def get_description(self) -> str:
        return "Signal-to-Noise Ratio (SNR) metric for audio, normalized to [0,1]"

    def _single_calculate(self, generated_item: Any, reference_item: Any, **kwargs: Any) -> float:
        """
        Calculate normalized SNR for a single pair of audio signals.

        The method loads the audio, ensures sample rates and lengths match,
        computes the SNR in dB, and normalizes it to the [0, 1] range.

        Parameters
        ----------
        generated_item : Any
            The generated audio signal (e.g., path or numpy array).
        reference_item : Any
            The reference audio signal.
        kwargs : Any
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The normalized SNR score, between 0.0 and 1.0.
        """
        gen_audio, gen_sr = self._load_audio(generated_item)
        ref_audio, ref_sr = self._load_audio(reference_item)

        if gen_sr != ref_sr:
            warnings.warn(
                f"Sample rates differ (generated: {gen_sr}, reference: {ref_sr}). Resampling generated audio to match reference rate {ref_sr} Hz."
            )
            gen_audio = self._resample_audio(gen_audio, orig_sr=gen_sr, target_sr=ref_sr)

        gen_audio, ref_audio = self._ensure_same_length(gen_audio, ref_audio)
        noise = gen_audio - ref_audio
        signal_power = np.mean(np.square(ref_audio)) + self.epsilon
        noise_power = np.mean(np.square(noise)) + self.epsilon
        snr_db = 10 * np.log10(signal_power / noise_power)
        normalized = (snr_db - self.snr_min) / (self.snr_max - self.snr_min)
        return float(np.clip(normalized, 0.0, 1.0))


class AudioSpectrogramDistance(AudioMetric):
    """
    Calculates similarity based on the distance between audio spectrograms.

    This metric is effective for capturing differences in frequency content
    (timbre, harmonics) over time. It computes the Short-Time Fourier Transform
    (STFT) for both signals, calculates a distance between the resulting
    spectrograms, and converts this distance to a similarity score from 0 to 1.
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
        Initialize the AudioSpectrogramDistance metric.

        Parameters
        ----------
        n_fft : int, optional
            The length of the FFT window, determining frequency resolution.
            Defaults to 2048.
        hop_length : int, optional
            The number of samples between successive FFT windows, determining
            time resolution. Defaults to 512.
        distance_type : str, optional
            The type of distance to calculate between spectrograms.
            Options: "euclidean", "cosine", "correlation". Defaults to "euclidean".
        window : str, optional
            The window function to apply to each FFT frame.
            Defaults to "hann".
        sample_rate : int, optional
            Target sample rate for audio processing. Defaults to None.
        kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If an unsupported `distance_type` is provided.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        valid_distances = ["euclidean", "cosine", "correlation"]
        if distance_type not in valid_distances:
            raise ValueError(
                f"Invalid distance_type '{distance_type}'. Must be one of: {valid_distances}"
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.distance_type = distance_type
        self.window = window

    def get_name(self) -> str:
        return f"AudioSpectrogramDistance_{self.distance_type}"

    def get_description(self) -> str:
        return f"Spectrogram-based distance metric for audio using {self.distance_type} distance"

    def _compute_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute the magnitude spectrogram for an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            The 1D audio signal.
        sr : int
            The sample rate of the audio.

        Returns
        -------
        np.ndarray
            The magnitude spectrogram as a 2D numpy array.

        Raises
        ------
        ImportError
            If `scipy` is required but not installed.
        ValueError
            If the audio signal is too short for the given `n_fft`.
        """
        if not _scipy_signal:
            raise ImportError("Scipy is required for spectrogram computation.")

        if len(audio) < self.n_fft:
            raise ValueError(
                f"Audio signal is too short ({len(audio)} samples) for the given n_fft ({self.n_fft})."
            )
        _, _, Zxx = _scipy_signal.stft(
            audio,
            fs=sr,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return np.abs(Zxx)

    def _calculate_distance(self, spec1: np.ndarray, spec2: np.ndarray) -> float:
        """
        Calculate the distance between two spectrograms.

        Parameters
        ----------
        spec1 : np.ndarray
            The first spectrogram.
        spec2 : np.ndarray
            The second spectrogram.

        Returns
        -------
        float
            The calculated distance.
        """
        min_frames = min(spec1.shape[1], spec2.shape[1])
        spec1 = spec1[:, :min_frames]
        spec2 = spec2[:, :min_frames]
        spec1_flat = spec1.flatten()
        spec2_flat = spec2.flatten()

        if self.distance_type == "euclidean":
            dist = np.linalg.norm(spec1_flat - spec2_flat)
            norm_factor = (np.linalg.norm(spec1_flat) + np.linalg.norm(spec2_flat)) / 2 + 1e-10
            return dist / norm_factor
        elif self.distance_type == "cosine":
            norm1 = np.linalg.norm(spec1_flat)
            norm2 = np.linalg.norm(spec2_flat)
            if norm1 == 0 or norm2 == 0:
                return 0.0 if norm1 == norm2 else 1.0
            sim = np.dot(spec1_flat, spec2_flat) / (norm1 * norm2)
            return 1.0 - np.clip(sim, -1.0, 1.0)
        elif self.distance_type == "correlation":
            if np.std(spec1_flat) == 0 or np.std(spec2_flat) == 0:
                return 0.0 if np.array_equal(spec1_flat, spec2_flat) else 1.0
            corr = np.corrcoef(spec1_flat, spec2_flat)[0, 1]
            return 1.0 - np.clip(corr, -1.0, 1.0)
        return 1.0

    def _single_calculate(self, generated_item: Any, reference_item: Any, **kwargs: Any) -> float:
        """
        Calculate spectrogram distance for a single pair of audio signals.

        The method computes spectrograms for both audio items, calculates the
        specified distance between them, and converts this distance to a
        similarity score between 0 and 1.

        Parameters
        ----------
        generated_item : Any
            The generated audio signal (e.g., path or numpy array).
        reference_item : Any
            The reference audio signal.
        kwargs : Any
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The spectrogram-based similarity score, between 0.0 and 1.0.
        """
        gen_audio, gen_sr = self._load_audio(generated_item)
        ref_audio, ref_sr = self._load_audio(reference_item)

        if gen_sr != ref_sr:
            warnings.warn(
                f"Sample rates differ (generated: {gen_sr}, reference: {ref_sr}). Resampling generated audio to match reference rate {ref_sr} Hz."
            )
            gen_audio = self._resample_audio(gen_audio, orig_sr=gen_sr, target_sr=ref_sr)

        gen_spec = self._compute_spectrogram(gen_audio, gen_sr)
        ref_spec = self._compute_spectrogram(ref_audio, ref_sr)
        distance = self._calculate_distance(gen_spec, ref_spec)
        similarity = np.exp(-distance)
        return float(np.clip(similarity, 0.0, 1.0))
