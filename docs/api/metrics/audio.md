# Audio Metrics

This section covers metrics for evaluating audio signals, supporting various input formats including numpy arrays, file paths, and raw audio data. These metrics are particularly useful for evaluating audio generation models, speech synthesis systems, and general audio processing applications.

::: gaico.metrics.audio.AudioSNRNormalized

The `AudioSNRNormalized` metric provides a normalized Signal-to-Noise Ratio (SNR) comparison between generated and reference audio signals. Unlike traditional SNR measurements that can range from negative infinity to positive infinity, this metric normalizes the result to a [0, 1] range for consistency with other GAICo metrics, where 1.0 indicates perfect audio quality (identical signals) and 0.0 indicates maximum noise/distortion.

### Input Format

The metric accepts various audio input formats, providing flexibility for different use cases:

- **NumPy Arrays**: Direct audio waveform data as `np.ndarray` (mono or stereo)
- **File Paths**: String paths to audio files (`.wav`, `.mp3`, `.flac`, etc.)
- **Python Lists/Tuples**: Raw audio samples as Python sequences
- **Mixed Formats**: Generated and reference can use different input formats

**Example Input Formats**:
```python
# NumPy array (recommended for programmatic use)
audio_array = np.array([0.1, 0.2, -0.1, 0.3, ...], dtype=np.float32)

# File path (convenient for stored audio)
audio_file = "/path/to/audio.wav"

# Python list (simple but less efficient)
audio_list = [0.1, 0.2, -0.1, 0.3, 0.0, -0.2]

# Stereo audio (automatically converted to mono)
stereo_audio = np.array([[0.1, 0.15], [0.2, 0.25], ...])  # Shape: (samples, 2)
```

### Calculation

The AudioSNRNormalized metric follows a multi-step process to ensure robust and meaningful comparisons:

1.  **Audio Loading and Preprocessing**:
   - Load audio from various input formats using `soundfile`
   - Convert stereo audio to mono using mean averaging: `mono = np.mean(stereo, axis=1)`
   - Handle sample rate mismatches through resampling using scipy.signal.resample
   - Ensure both signals have the same length by truncating to the shorter duration

2. **Noise Calculation**:
   - Compute noise as the difference between generated and reference signals: `noise = generated - reference`
   - This treats the reference as the "clean" signal and generated as "noisy"

3. **SNR Computation**:
   - Calculate signal power: `signal_power = np.mean(reference²) + epsilon`
   - Calculate noise power: `noise_power = np.mean(noise²) + epsilon`
   - Compute SNR in decibels: `SNR_dB = 10 * log₁₀(signal_power / noise_power)`
   - Apply epsilon (default 1e-10) to prevent division by zero

4. **Normalization**:
   - Map SNR from decibel range to [0, 1] using linear scaling with clipping
   - Formula: `normalized_score = (SNR_dB - SNR_min) / (SNR_max - SNR_min)`
   - Default range: SNR_min = -20 dB (very noisy), SNR_max = 40 dB (very clean)
   - Apply clipping: `score = max(0.0, min(1.0, normalized_score))`

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

The metric accepts the same diverse input formats as `AudioSNRNormalized`, with additional considerations for spectral analysis:

- **Audio Duration**: Longer audio clips provide more reliable spectral analysis
- **Sample Rate Consistency**: While automatic resampling is supported, consistent sample rates yield better results
- **Minimum Length**: Very short audio clips (< 100 samples) may produce unreliable spectrograms

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

1. **Spectrogram Computation**:
   - Apply STFT using scipy.signal.stft with configurable parameters
   - Default window: Hann window (n_fft=2048 samples ≈ 46ms at 44.1kHz)
   - Default hop length: 512 samples (75% overlap, good time-frequency resolution)
   - Extract magnitude spectrogram: `magnitude = |STFT(audio)|`

2. **Temporal Alignment**:
   - Ensure spectrograms have matching time dimensions by truncating to shorter length
   - Handle edge case where one spectrogram is empty (zero frames)

3. **Distance Calculation** (configurable via `distance_type`):

   **Euclidean Distance** (default):
   - Compute normalized mean squared difference: `distance = sqrt(mean((spec1 - spec2)²))`
   - Normalize by average magnitude to handle scale differences
   - Good for overall spectral energy comparison

   **Cosine Distance**:
   - Flatten spectrograms and compute cosine similarity: `cos_sim = dot(spec1, spec2) / (||spec1|| × ||spec2||)`
   - Convert to distance: `distance = 1 - cos_sim`
   - Effective for comparing spectral patterns regardless of overall energy

   **Correlation Distance**:
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
