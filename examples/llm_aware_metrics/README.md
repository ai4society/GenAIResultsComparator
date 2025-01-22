# LLM-Aware Metrics

This module extends the base LLM-Metrics library to provide LLM-specific evaluation capabilities by incorporating prompts and metadata into the comparison process.

The metrics for this module are designed to be used in conjunction with the existing metrics from the base library, enhancing their functionality to better suit LLM outputs. However, they now take into account the prompts and metadata associated with the LLM outputs, providing a more nuanced evaluation.

* **_Inputs:_** text 1, text 2, prompt 1, prompt 2 (optional), metadata (optional)
* **_Output:_** similarity score or validation result

The new base class for this model is `LLMAwareMetric` (under `code/base.py`), which can be extended to create specific metrics like:
* `PromptAwareMetric` (under `code/prompt_aware.py`)
* `SchemaAwareMetric` (under `code/schema_based.py`)
* `AggregatedSimilarityMetric` (under `code/aggregated_similarity_score.py`).


## Quick Start

```python
from llm_metrics import BLEU
from examples.llm_aware_metrics.code.prompt_aware import PromptAwareMetric

# Initialize the metric
base_metric = BLEU()
prompt_aware = PromptAwareMetric(base_metric)

# Compare two responses with their prompts
prompt1 = "Summarize the following text:"
response1 = "This is a summary."
prompt2 = "Create a brief summary:"
response2 = "This is another summary."

score = prompt_aware.calculate_with_prompt(
    text1=response1,
    text2=response2,
    prompt1=prompt1,
    prompt2=prompt2
)
```

## Available Metrics
All metrics from the base library can be used with the LLM-aware metrics. The following LLM-specific metrics are available (please check the `code` directory for the implementation details):

### PromptAwareMetric
Compares LLM outputs while considering the prompts that generated them. It is suitable for general LLM output comparison

Currently, the prompts are prepended to the responses (by a space) before comparison using the supplied base metric.

### SchemaAwareMetric
Validates and compares structured outputs against predefined schemas
   - Validates outputs against JSON schemas
   - Ensures structural consistency
   - Ideal for comparing structured outputs

### AggregatedSimilarityMetric
AggregatedSimilarityMetric measures three key aspects:
- How similar Response 1 is to Prompt 1
- How similar Response 2 is to Prompt 2
- How similar Response 1 is to Response 2

The idea is that two responses should not only be similar to each other but should also be similar with their respective prompts.
The 3 similarity comparisons can be implemented with any base metric from the `llm_metric` library, and the scores are aggregated to provide a comprehensive similarity score.

## Installation

The module is part of the examples in the LLM-Metrics library. No additional installation is required if you have the main library installed.

## Example Notebooks

1. `example_prompt_aware_comparison.ipynb`: Basic usage and prompt-aware comparison
2. `example_schema_based_comparison.ipynb`: Working with structured outputs
3. `example_agg_similarity_score.ipynb`: Advanced aggregation scoring examples


## Usage Guidelines

1. **Choosing the Right Metric**
   - Use `PromptAwareMetric` for general comparison tasks
   - Use `SchemaAwareMetric` when working with structured outputs
   - Use `AggregatedSimilarityMetric` when prompt adherence is crucial

2. **Working with Metadata**
   - Pass schemas through metadata for `SchemaAwareMetric`
   - Include task-specific parameters in metadata
   - Use metadata for custom scoring adjustments

3. **Batch Processing**
   - Use batch methods for multiple comparisons
   - Ensure consistent prompt-response pairing
   - Consider using metadata for batch-wide parameters
