# Audio Metrics

This section covers metrics for evaluating audio signals, supporting various input formats including numpy arrays, file paths, and raw audio data. These metrics are particularly useful for evaluating audio generation models, speech synthesis systems, and general audio processing applications.

::: gaico.metrics.audio.AudioSNRNormalized

The `AudioSNRNormalized` metric provides a normalized Signal-to-Noise Ratio (SNR) comparison between generated and reference audio signals. Unlike traditional SNR measurements that can range from negative infinity to positive infinity, this metric normalizes the result to a [0, 1] range for consistency with other GAICo metrics, where 1.0 indicates perfect audio quality (identical signals) and 0.0 indicates maximum noise/distortion.

### Input Format

The metric accepts various audio input formats, providing flexibility for different use cases. It intelligently handles both single-item and batch comparisons.

- **Single Item Formats**:
  - **NumPy Array**: A 1D `np.ndarray` representing a mono audio waveform.
  - **File Path**: A string path to an audio file (e.g., `.wav`, `.flac`).
- **Batch Item Formats**:
  - **List/Tuple/Pandas Series**: An iterable where each element is a single audio item (either a path or a 1D array).
  - **2D NumPy Array**: An array where each row is treated as a separate audio signal.
- **Mixed Formats**: Generated and reference inputs can use different formats (e.g., comparing a file path to a NumPy array).
- **Audio Preprocessing**:
  - **Stereo to Mono**: Stereo signals (2D arrays or files) are automatically converted to mono by averaging the channels.
  - **Resampling**: If sample rates differ, the generated audio is resampled to match the reference audio's rate.

### Error Handling

The metric is designed to be robust and will raise specific errors for invalid inputs:

- `FileNotFoundError`: If a string path to an audio file does not exist.
- `TypeError`: If an unsupported data type (e.g., a dictionary) is provided as input.
- `ValueError`: If an audio array or list is empty, or if an audio file cannot be read.

### Calculation

The AudioSNRNormalized metric follows a multi-step process to ensure robust and meaningful comparisons:

1.  **Audio Loading and Preprocessing**:
    - Load audio from the specified input format (path, array, etc.).
    - If audio is stereo, convert it to mono by averaging the channels.
    - If sample rates differ, resample the generated audio to match the reference rate.
    - Ensure both signals have the same length by truncating the longer one.

2.  **Noise Calculation**:
    - Compute noise as the element-wise difference between the signals: `noise = generated - reference`.

3.  **SNR Computation**:
    - Calculate the power of the reference signal and the noise signal.
    - Compute the SNR in decibels (dB): `SNR_dB = 10 * log₁₀(signal_power / noise_power)`.
    - An epsilon value is added to prevent division by zero.

4.  **Normalization**:
    - Linearly scale the SNR (dB) to a [0, 1] range using configurable `snr_min` and `snr_max` values.
    - Clip the final score to ensure it falls strictly within the [0, 1] range.

### Usage

```python
from gaico.metrics.audio import AudioSNRNormalized
import numpy as np

# Initialize with default parameters
snr_metric = AudioSNRNormalized()

# Example 1: Compare numpy arrays (typical programmatic use)
# Generate a clean sine wave
t = np.linspace(0, 1, 44100)  # 1 second at 44.1 kHz
clean_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone

# Add some noise to create a "generated" signal
noise = 0.05 * np.random.randn(44100).astype(np.float32)
noisy_signal = clean_signal + noise

score = snr_metric.calculate(noisy_signal, clean_signal)
print(f"SNR Score: {score:.3f}")
# Expected output: SNR Score: 0.856 (high score due to low noise)

# Example 2: Compare audio files (typical file-based workflow)
generated_file = "path/to/generated_speech.wav"
reference_file = "path/to/reference_speech.wav"

score = snr_metric.calculate(generated_file, reference_file)
print(f"File-based SNR Score: {score:.3f}")

# Example 3: Custom SNR range for specific application
# For speech synthesis, you might want different thresholds
speech_snr_metric = AudioSNRNormalized(
    snr_min=-10.0,  # Acceptable speech quality lower bound
    snr_max=30.0,   # High-quality speech upper bound
    sample_rate=16000  # Common speech sampling rate
)

# Example 4: Batch processing multiple audio comparisons
generated_audios = [noisy_signal, noisy_signal * 0.5, clean_signal]
reference_audios = [clean_signal, clean_signal, clean_signal]

batch_scores = snr_metric.calculate(generated_audios, reference_audios)
print(f"Batch SNR Scores: {[f'{s:.3f}' for s in batch_scores]}")
# Expected output: ['0.856', '0.923', '1.000']
```

---

::: gaico.metrics.audio.AudioSpectrogramDistance

The `AudioSpectrogramDistance` metric evaluates audio similarity by comparing spectrograms (frequency-time representations) rather than raw waveforms. This approach is particularly effective for capturing timbral and spectral characteristics, making it suitable for evaluating music generation, speech synthesis, and audio effects processing where frequency content matters more than exact waveform matching.

### Input Format

The metric accepts the same diverse input formats as `AudioSNRNormalized` (file paths, NumPy arrays, lists, etc.) and follows the same preprocessing and error handling logic. Additional considerations for spectral analysis include:

- **Audio Duration**: Longer audio clips provide more reliable spectral analysis.
- **Sample Rate Consistency**: While automatic resampling is supported, providing signals with consistent sample rates yields the most accurate results.
- **Minimum Length**: Very short audio clips (e.g., shorter than the `n_fft` size) will raise a `ValueError` as a reliable spectrogram cannot be computed.

**Recommended Input Characteristics**:
```python
# Optimal for spectral analysis (at least 0.1 seconds of audio)
optimal_length = int(0.1 * sample_rate)  # 4410 samples at 44.1 kHz
audio_signal = np.random.randn(optimal_length)

# Music/speech applications typically use these sample rates
sample_rates = {
    'speech': 16000,      # Common for speech processing
    'music': 44100,       # CD quality
    'professional': 48000  # Professional audio
}
```

### Calculation

The AudioSpectrogramDistance metric employs Short-Time Fourier Transform (STFT) analysis followed by distance computation:

1.  **Spectrogram Computation**:
    - Check if `scipy` is available, raising an `ImportError` if not.
    - Apply STFT using `scipy.signal.stft` with configurable parameters (`n_fft`, `hop_length`, `window`).
    - Extract the magnitude spectrogram: `magnitude = |STFT(audio)|`.

2.  **Temporal Alignment**:
    - Ensure spectrograms have matching time dimensions by truncating the longer one to match the shorter one.

3.  **Distance Calculation** (configurable via `distance_type`):

    **Euclidean Distance** (default):
    - Computes the Euclidean distance between the flattened spectrograms and normalizes it by the average magnitude to handle scale differences.

    **Cosine Distance**:
   - Compute normalized cross-correlation after mean removal
   - Formula: `correlation = dot(spec1_centered, spec2_centered) / (||spec1_centered|| × ||spec2_centered||)`
   - Distance: `distance = 1 - correlation`
   - Captures linear relationships in spectral content

4. **Similarity Conversion**:
   - Transform distance to similarity using exponential decay: `similarity = exp(-distance)`
   - Maps distance [0, ∞) to similarity (1, 0], where 1.0 indicates identical spectrograms
   - Apply clipping to ensure [0, 1] range

### Usage

```python
from gaico.metrics.audio import AudioSpectrogramDistance
import numpy as np

# Initialize with default parameters (Euclidean distance)
spec_metric = AudioSpectrogramDistance()

# Example 1: Compare harmonic content
# Generate two sine waves with different frequencies
t = np.linspace(0, 1, 44100)
signal_440hz = np.sin(2 * np.pi * 440 * t)  # A4 note
signal_880hz = np.sin(2 * np.pi * 880 * t)  # A5 note (octave higher)
signal_440hz_copy = signal_440hz + 0.01 * np.random.randn(44100)  # Slight noise

score_identical = spec_metric.calculate(signal_440hz, signal_440hz_copy)
score_different = spec_metric.calculate(signal_440hz, signal_880hz)

print(f"Similar frequency content: {score_identical:.3f}")  # ~0.95-0.99
print(f"Different frequency content: {score_different:.3f}")  # ~0.60-0.80

# Example 2: Music analysis with custom parameters
# Optimized for music: longer window for better frequency resolution
music_metric = AudioSpectrogramDistance(
    n_fft=4096,           # Higher frequency resolution
    hop_length=1024,      # 75% overlap maintained
    distance_type="cosine", # Pattern-based comparison
    window="blackman"     # Reduced spectral leakage
)

# Example 3: Speech analysis with appropriate parameters
# Optimized for speech: shorter window for better time resolution
speech_metric = AudioSpectrogramDistance(
    n_fft=1024,           # ~64ms window at 16kHz
    hop_length=256,       # 75% overlap
    distance_type="correlation",
    sample_rate=16000     # Common speech sample rate
)

# Example 4: Compare different distance types
metrics_comparison = {
    'euclidean': AudioSpectrogramDistance(distance_type="euclidean"),
    'cosine': AudioSpectrogramDistance(distance_type="cosine"),
    'correlation': AudioSpectrogramDistance(distance_type="correlation")
}

# Complex signal with harmonics
fundamental = 220  # A3
complex_signal = (
    np.sin(2 * np.pi * fundamental * t) +           # Fundamental
    0.5 * np.sin(2 * np.pi * 2 * fundamental * t) + # 2nd harmonic
    0.25 * np.sin(2 * np.pi * 3 * fundamental * t)  # 3rd harmonic
)

# Signal with shifted harmonics (different timbre)
shifted_signal = (
    0.7 * np.sin(2 * np.pi * fundamental * t) +
    0.6 * np.sin(2 * np.pi * 2 * fundamental * t) +
    0.4 * np.sin(2 * np.pi * 4 * fundamental * t)  # 4th instead of 3rd harmonic
)

for name, metric in metrics_comparison.items():
    score = metric.calculate(complex_signal, shifted_signal)
    print(f"{name.capitalize()} distance score: {score:.3f}")

# Example 5: Batch processing for audio dataset evaluation
generated_samples = [signal_440hz, signal_880hz, complex_signal]
reference_samples = [signal_440hz_copy, signal_440hz, shifted_signal]

batch_scores = spec_metric.calculate(generated_samples, reference_samples)
print(f"Batch spectrogram scores: {[f'{s:.3f}' for s in batch_scores]}")
```
