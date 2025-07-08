# Time Series Metrics

This section covers metrics for evaluating time-series data, particularly when represented as textual or structured sequences.

::: gaico.metrics.structured.TimeSeriesElementDiff

The `TimeSeriesElementDiff` metric provides a weighted comparison between two time series. It evaluates similarity based on both the presence of common time points (keys) and the closeness of their corresponding values. The metric allows you to weigh the importance of a matching key more heavily than a matching value.

### Input Format

The metric expects time-series data as strings, where each time point and its value are represented as a "key:value" pair, with pairs separated by commas.

- **Example Generated Time Series**: `"t1:100, t2:105, t4:110"`
- **Example Reference Time Series**: `"t1:102, t3:120, t4:110"`

Values are expected to be numeric (float). Malformed pairs (e.g., missing colon, non-numeric value) or duplicate keys are handled during parsing, with warnings issued.

### Calculation

1.  **Initialization**: The metric can be initialized with a `key_to_value_weight_ratio` (default is `2.0`), which sets the weight of a key match relative to a perfect value match. For a ratio of 2, the key weight is `2.0` and the value weight is `1.0`.
2.  **Parsing**: Both the generated and reference strings are parsed into dictionaries mapping keys (e.g., `'t1'`) to their float values (e.g., `100.0`).
3.  **Comparison**: The metric iterates over the union of all keys from both time series. For each key, a score is calculated and compared against the maximum possible score for that key (`key_weight + value_weight`).
    - If a key exists in **both** series:
      - The score gets `key_weight` for the key match.
      - A value similarity score is calculated as `max(0, 1 - |v_gen - v_ref| / |v_ref|)`. This score is `1.0` for a perfect match and decreases towards `0.0` as the relative difference grows.
      - The score gets `value_weight * value_similarity`.
    - If a key exists in **only one** of the series, it contributes `0` to the total score.
4.  **Normalization**: The final score is the sum of all accumulated scores divided by the sum of all maximum possible scores.
    - `Score = Total_Accumulated_Score / Total_Max_Possible_Score`
    - If both series are empty, the score is `1.0`.

The final score is a float between `0.0` and `1.0`, where `1.0` indicates a perfect match in both keys and values.

### Usage

```python
from gaico.metrics.structured import TimeSeriesElementDiff

# Initialize with default key_weight=2.0, value_weight=1.0
metric = TimeSeriesElementDiff()

generated_ts = "t1:100, t2:105, t4:110"
reference_ts = "t1:102, t3:120, t4:110"

# --- Manual Calculation ---
# Keys union: {t1, t2, t3, t4}. Max score per key = 2+1=3. Total max score = 4*3=12.
# t1: In both. Key score=2. Value sim = 1 - |100-102|/102 ≈ 0.98. Value score = 1*0.98. Total=2.98
# t2: Only in generated. Total=0
# t3: Only in reference. Total=0
# t4: In both. Key score=2. Value sim = 1 - |110-110|/110 = 1.0. Value score = 1*1.0. Total=3.0
# Total accumulated score = 2.98 + 0 + 0 + 3.0 = 5.98
# Final score = 5.98 / 12 ≈ 0.498

score = metric.calculate(generated_ts, reference_ts)
print(f"TimeSeriesElementDiff Score: {score:.3f}")
# Expected output: TimeSeriesElementDiff Score: 0.498
```

---

::: gaico.metrics.structured.TimeSeriesDTW

The `TimeSeriesDTW` metric calculates the similarity between two time series using Dynamic Time Warping (DTW). DTW is particularly effective for measuring the similarity between two temporal sequences that may vary in speed or have phase differences. It finds the optimal alignment between the sequences, and the DTW distance is the sum of distances between the aligned points.

### Input Format

The metric accepts the same input format as `TimeSeriesElementDiff`: a string of comma-separated "key:value" pairs or simple values. However, **`TimeSeriesDTW` only uses the values for its calculation**, preserving their original order and ignoring the keys. This allows for direct comparison of the value sequences.

- **Example Generated Time Series**: `"t1:1, t2:2, t3:3, t4:4, t5:3, t6:2"` (parsed as `[1, 2, 3, 4, 3, 2]`)
- **Example Reference Time Series**: `"a:1, b:2.2, c:3.1, d:4, e:3.5"` (parsed as `[1, 2.2, 3.1, 4, 3.5]`)
- **Mixed Format**: `"t1:1, 2.2, 3.1, t4:4"` (parsed as `[1, 2.2, 3.1, 4]`)

Any non-numeric parts in the value will be ignored, and a warning will be issued.

### Calculation

1.  **Dependency Check**: This metric requires the `dtaidistance` library which is pre-installed with `gaico`. If it's not installed, an `ImportError` will be raised.
2.  **Parsing**: Both the generated and reference strings are parsed into NumPy arrays of floating-point numbers, extracting only the values in their original order.
3.  **DTW Distance**: The `dtaidistance.dtw.distance` function is used to compute the raw DTW distance between the two value sequences.
4.  **Normalization**: The raw distance is not bounded and depends on the scale of the values and length of the sequences. To convert it into a normalized similarity score between 0 and 1, the following formula is used:
    - `Score = 1.0 / (1.0 + DTW_Distance)`
    - If the DTW distance is `0` (indicating identical sequences), the score is `1.0`.
    - As the distance increases, the score approaches `0.0`.
    - If both sequences are empty, the score is `1.0`. If only one is empty, the score is `0.0`.

### Usage

```python
from gaico.metrics.structured import TimeSeriesDTW

# Initialize the metric
metric = TimeSeriesDTW()

# Two similar time series, slightly out of phase. Keys are ignored.
generated_ts = "t1:1, t2:2, t3:3, t4:4, t5:3, t6:2"
reference_ts = "a:1, b:2.2, c:3.1, d:4, e:3.5"

# The metric will extract the value sequences:
# Generated values: [1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
# Reference values: [1.0, 2.2, 3.1, 4.0, 3.5]
# The dtaidistance library will calculate the optimal alignment distance.

score = metric.calculate(generated_ts, reference_ts)
print(f"TimeSeriesDTW Score: {score:.3f}")
# Expected output might be around: TimeSeriesDTW Score: 0.385

# Identical value sequences, different keys
generated_ts_2 = "k1:5, k2:6, k3:7"
reference_ts_2 = "z1:5, z2:6, z3:7"

# DTW distance will be 0 because value sequences are identical.
# Score = 1 / (1 + 0) = 1.0
score_2 = metric.calculate(generated_ts_2, reference_ts_2)
print(f"TimeSeriesDTW Score (identical values): {score_2:.3f}")
# Expected output: TimeSeriesDTW Score (identical values): 1.000
```
