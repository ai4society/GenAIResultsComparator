# Time Series Metrics

This section covers metrics for evaluating time-series data, particularly when represented as textual or structured sequences.

## TimeSeriesElementDiff

The `TimeSeriesElementDiff` metric compares two time series based on the presence and absence of their time points (or keys). It does not consider the values associated with these time points in the current version, focusing instead on the structural similarity of the time point sets.

### Input Format

The metric expects time-series data as strings, where each time point and its value are represented as a "key:value" pair, with pairs separated by commas.

- **Example Generated Time Series**: `"t1:100, t2:105, t4:110"`
- **Example Reference Time Series**: `"t1:100, t3:102, t4:110"`

Values are expected to be numeric (float). Malformed pairs (e.g., missing colon, non-numeric value) are skipped during parsing with a warning.

### Calculation

1.  **Parsing**: Both the generated and reference strings are parsed into lists of `(key, value)` tuples. Only the keys are used for this metric.
    - `"t1:70, t2:72, t3:75"` results in keys `{'t1', 't2', 't3'}`.
2.  **Comparison**: The metric calculates the Jaccard similarity index between the set of keys from the generated time series and the set of keys from the reference time series.
    - `Jaccard_Index = |Keys_Generated ∩ Keys_Reference| / |Keys_Generated ∪ Keys_Reference|`
3.  **Normalization**: The Jaccard index is naturally a score between `0.0` and `1.0`.
    - If both time series parse to an empty set of keys (e.g., empty input strings), the score is `1.0`.
    - If one set of keys is empty and the other is not, the score is `0.0`.

The final score is a float between `0.0` and `1.0`, where `1.0` indicates that both time series have the exact same set of time points, and `0.0` indicates no common time points (when at least one series has time points).

### Usage

```python
from gaico.metrics.structured import TimeSeriesElementDiff

metric = TimeSeriesElementDiff()

generated_ts = "t_1: 70, t_2: 72, t_3: 75"
reference_ts = "t_1: 70, t_3: 70, t_5: 80" # t_2 missing, t_5 extra

# Generated Keys: {t_1, t_2, t_3}
# Reference Keys: {t_1, t_3, t_5}
# Intersection: {t_1, t_3} (size 2)
# Union: {t_1, t_2, t_3, t_5} (size 4)
# Score = 2 / 4 = 0.5

score = metric.calculate(generated_ts, reference_ts)
print(f"TimeSeriesElementDiff Score: {score}")
# Expected output: TimeSeriesElementDiff Score: 0.5
```
