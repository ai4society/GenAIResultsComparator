# GAICo Examples

This directory contains example notebooks demonstrating various use cases of the GAICo library. Each example is designed to showcase practical applications and provide detailed guidance on using different metrics for specific scenarios.

## Examples

### 1. `quickstart.ipynb`

- Using GAICo's `Experiment` module to provide a simple, quickstart workflow.

### 2. `example-1.ipynb`: Multiple Models, Single Metric

- Evaluating multiple models (LLMs, Google, and Custom) using a single metric.

### 3. `example-2.ipynb`: Single Model, Multiple Metric

- Evaluating a single model on multiple metrics.

### 4. `DeepSeek-example.ipynb`: Testing _DeepSeek R1_

- The aim for this notebook was to aid with evaluating DeepSeek R1 for [AI4Society's Point of View (POV)](https://drive.google.com/file/d/1ErR1xT7ftvmHiUyYrdUbjyd4qCK_FxKX/view?usp=sharing).
- **Note**: All results remove the `<think>` tags for the DeepSeek models.

## Advanced Examples

The `advanced-examples` directory contains advances notebooks showcasing more complex use cases and metrics. These examples are intended for users who are already familiar with the basics of GAICo. Please refer to the README.md file in that directory for details. A quick description:

### 1. `llm_faq-example.ipynb`: LLM FAQ Analysis

- Comparison of various LLM responses (Phi, Mixtral, etc.) on FAQ dataset from USC.

### 2. `threshold-example.ipynb`: Thresholds

- Exploration of default and custom thresholding techniques for LLM responses.

### 3. `viz-example.ipynb`: Visualizations

- Hands-on visualizations for LLM results.
