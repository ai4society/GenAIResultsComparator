# Planning Metrics

This section details metrics specialized for evaluating outputs in automated planning, typically sequences of actions.

## ActionSequenceDiff

The `ActionSequenceDiff` metric evaluates the similarity between two action sequences. It is designed for outputs common in automated planning where an LLM might generate a sequence of actions to achieve a goal.

### Input Format

The metric expects input sequences as strings, where actions are comma-separated. Concurrent actions (actions that can happen in parallel or are part of a single step) can be grouped using curly braces `{}`.

- **Example Generated Sequence**: `"take(objA), move(loc1, loc2), {action_set_1(param), action_set_2}, drop(objA)"`
- **Example Reference Sequence**: `"take(objA), move(loc1, loc2), drop(objA)"`

Each action or action set is treated as a single element in the sequence during comparison. Empty elements resulting from parsing (e.g., from `,,` or empty strings in sets like `{,}`) are handled by the parser.

### Calculation

1.  **Parsing**: Both the generated and reference strings are parsed into lists of elements. Each element is either a string (for a single action) or a `frozenset` of strings (for a set of concurrent actions).
    - `"a1, {a2, a3}, a4"` becomes `['a1', frozenset({'a2', 'a3'}), 'a4']`.
2.  **Comparison**: The metric calculates the length of the Longest Common Subsequence (LCS) between the two parsed sequences.
3.  **Normalization**: The score is normalized by dividing the LCS length by the length of the longer of the two sequences.
    - `Score = LCS_Length / max(Length_Generated_Sequence, Length_Reference_Sequence)`
    - If both sequences are empty after parsing, the score is `1.0`. If one is empty and the other is not, the score is `0.0` (as LCS length would be 0).

The final score is a float between `0.0` and `1.0`, where `1.0` indicates identical sequences (or both empty) and `0.0` indicates no common subsequence elements (when at least one sequence is non-empty).

### Usage

```python
from gaico.metrics.structured import ActionSequenceDiff

metric = ActionSequenceDiff()

generated_plan = "pickup(A), stack(A,B), {noop1, noop2}, pickup(C)"
reference_plan = "pickup(A), stack(A,B), pickup(C)"

score = metric.calculate(generated_plan, reference_plan)
print(f"ActionSequenceDiff Score: {score}")
# Example output might be: ActionSequenceDiff Score: 0.75
# Generated: [pickup(A), stack(A,B), {noop1, noop2}, pickup(C)] (len 4)
# Reference: [pickup(A), stack(A,B), pickup(C)] (len 3)
# LCS: [pickup(A), stack(A,B), pickup(C)] (len 3)
# Score = 3 / max(4,3) = 3/4 = 0.75
```
