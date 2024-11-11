# LLM-Aware Metrics

This module extends the base LLM-Metrics library to provide LLM-specific evaluation capabilities by incorporating prompts and metadata into the comparison process.

## Features

- **Prompt-Aware Comparison**: Compare LLM outputs while considering the prompts that generated them
- **Schema-Based Validation**: Validate and compare structured outputs against predefined schemas
- **Aggregated Similarity Scoring**: Evaluate how well responses align with their prompts
- **Flexible Integration**: Works with all existing metrics from the base library

## Installation

The module is part of the examples in the LLM-Metrics library. No additional installation is required if you have the main library installed.

## Available Metrics
All metrics from the base library can be used with the LLM-aware metrics. The following LLM-specific metrics are available (please check the `code` directory for the implementation details):

1. **PromptAwareMetric**
   - Incorporates prompts into the comparison process
   - Uses any base metric from the main library
   - Suitable for general LLM output comparison

2. **SchemaAwareMetric**
   - Validates outputs against JSON schemas
   - Ensures structural consistency
   - Ideal for comparing structured outputs

3. **AggregatedSimilarityMetric**
   - Measures different aggregation levels of similarity within the prompts and responses


## Example Notebooks

1. `example_prompt_aware_comparison.ipynb`: Basic usage and prompt-aware comparison
2. `example_schema_based_comparison.ipynb`: Working with structured outputs
3. `example_agg_similarity_score.ipynb`: Advanced aggregation scoring examples

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
