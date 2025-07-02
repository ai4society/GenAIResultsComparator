# Planning Metrics

This section details metrics specialized for evaluating outputs in automated planning, typically sequences of actions.

::: gaico.metrics.structured.PlanningLCS

The `PlanningLCS` metric evaluates the similarity between two action sequences by respecting the order of actions. It is designed for outputs common in automated planning where an LLM might generate a sequence of actions to achieve a goal.

### Input Format

The metric expects input sequences as strings, where actions are comma-separated. Concurrent actions (actions that can happen in parallel or are part of a single step) can be grouped using curly braces `{}`.

- **Example Generated Sequence**: `"take(objA), move(loc1, loc2), {action_set_1(param), action_set_2}, drop(objA)"`
- **Example Reference Sequence**: `"take(objA), move(loc1, loc2), drop(objA)"`

Each action or action set is treated as a single element in the sequence during comparison.

### Calculation

1.  **Parsing**: Both the generated and reference strings are parsed into lists of elements. Each element is either a string (for a single action) or a `frozenset` of strings (for a set of concurrent actions).
    - `"a1, {a2, a3}, a4"` becomes `['a1', frozenset({'a2', 'a3'}), 'a4']`.
2.  **Comparison**: The metric calculates the length of the Longest Common Subsequence (LCS) between the two parsed sequences.
3.  **Normalization**: The score is normalized by dividing the LCS length by the length of the longer of the two sequences.
    - `Score = LCS_Length / max(Length_Generated_Sequence, Length_Reference_Sequence)`
    - If both sequences are empty after parsing, the score is `1.0`.

The final score is a float between `0.0` and `1.0`, where `1.0` indicates identical sequences.

### Usage

```python
from gaico.metrics.structured import PlanningLCS

metric = PlanningLCS()

generated_plan = "pickup(A), stack(A,B), {noop1, noop2}, pickup(C)"
reference_plan = "pickup(A), stack(A,B), pickup(C)"

# Generated (parsed): ['pickup(A)', 'stack(A,B)', frozenset({'noop1', 'noop2'}), 'pickup(C)'] (len 4)
# Reference (parsed): ['pickup(A)', 'stack(A,B)', 'pickup(C)'] (len 3)
# LCS: ['pickup(A)', 'stack(A,B)', 'pickup(C)'] (len 3)
# Score = 3 / max(4, 3) = 3 / 4 = 0.75

score = metric.calculate(generated_plan, reference_plan)
print(f"PlanningLCS Score: {score}")
# Expected output: PlanningLCS Score: 0.75
```

::: gaico.metrics.structured.PlanningJaccard

The `PlanningJaccard` metric calculates the similarity between two action sequences by comparing their sets of unique actions, ignoring order and frequency. This is useful for checking if the same overall actions were taken, regardless of the sequence.

### Input Format

The input format is the same as for `PlanningLCS`. Concurrent actions in curly braces are flattened and treated as individual actions within the set.

### Calculation

1.  **Parsing**: Both sequences are parsed into lists of elements, just like in `PlanningLCS`.
2.  **Flattening**: The parsed lists are converted into flat sets of unique action strings.
    - `['a1', frozenset({'a2', 'a3'}), 'a4']` becomes the set `{'a1', 'a2', 'a3', 'a4'}`.
3.  **Comparison**: The metric calculates the Jaccard similarity index between the two sets of actions.
    - `Jaccard_Index = |Actions_Generated ∩ Actions_Reference| / |Actions_Generated ∪ Actions_Reference|`
4.  **Normalization**: The Jaccard index is naturally a score between `0.0` and `1.0`. If both sets are empty, the score is `1.0`.

### Usage

```python
from gaico.metrics.structured import PlanningJaccard

metric = PlanningJaccard()

generated_plan = "pickup(A), stack(A,B), pickup(C)"
reference_plan = "pickup(C), pickup(A), stack(A,B)" # Same actions, different order

# Generated Set: {'pickup(A)', 'stack(A,B)', 'pickup(C)'}
# Reference Set: {'pickup(C)', 'pickup(A)', 'stack(A,B)'}
# Intersection size: 3, Union size: 3
# Score = 3 / 3 = 1.0

score = metric.calculate(generated_plan, reference_plan)
print(f"PlanningJaccard Score (same actions, different order): {score}")
# Expected output: PlanningJaccard Score (same actions, different order): 1.0

generated_plan_2 = "pickup(A), {stack(A,B), noop}"
reference_plan_2 = "pickup(A), stack(A,B), drop(B)"

# Generated Set: {'pickup(A)', 'stack(A,B)', 'noop'}
# Reference Set: {'pickup(A)', 'stack(A,B)', 'drop(B)'}
# Intersection: {'pickup(A)', 'stack(A,B)'} (size 2)
# Union: {'pickup(A)', 'stack(A,B)', 'noop', 'drop(B)'} (size 4)
# Score = 2 / 4 = 0.5

score_2 = metric.calculate(generated_plan_2, reference_plan_2)
print(f"PlanningJaccard Score (different actions): {score_2}")
# Expected output: PlanningJaccard Score (different actions): 0.5
```

### Theoretical Background and Further Reading

The metrics used for comparing planning sequences in this library are inspired by foundational work on measuring diversity and similarity between plans. The core idea is that plans can be compared based on their constituent actions, the states they traverse, or their underlying causal structures.

The `PlanningJaccard` and `ActionSequenceDiff` metrics are direct implementations of **action-based, set-difference** measures discussed in the following papers. These measures are computationally efficient and provide a domain-independent way to quantify how different two plans are based on the actions they contain.

- Srivastava, Biplav, et al. "Finding inter-related plans." ICAPS 2006 (2006): 18.

- Srivastava, Biplav, et al. "Domain Independent Approaches for Finding Diverse Plans." IJCAI. 2007.

These papers formalize various distance functions (e.g., `δ1`, `δa`) that serve as the basis for our order-agnostic planning metrics.
