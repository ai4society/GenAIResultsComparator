# GAICo Examples

This directory contains example notebooks demonstrating various use cases of the GAICo library. Each example is designed to showcase practical applications and provide detailed guidance on using different metrics for specific scenarios.

## Table of Contents

- [Quickstart Examples](#quickstart-examples)
- [Basic Examples](#basic-examples)
- [Advanced Examples](#advanced-examples)
- [Citations](#citations)

## Quickstart Examples

The following notebooks demonstrate the

### 1. [`quickstart.ipynb`](quickstart.ipynb)

- Using GAICo's `Experiment` module to provide a simple, quickstart workflow.

### 2. [`example-1.ipynb`](example-1.ipynb): Multiple Models, Single Metric

- Evaluating multiple models (LLMs, Google, and Custom) using a single metric with `<metric>.calculate()` method.

### 3. [`example-2.ipynb`](example-2.ipynb): Single Model, Multiple Metric

- Evaluating a single model on multiple metrics with their `<metric>.calculate()` methods.

### 4. [`example-structured_data.ipynb`](example-structured_data.ipynb): Time-Series and Automated Planning

- A comparison of 'usual' text-based metrics with special metrics for time series and automated planning.

## Basic Examples

The `basic-examples` directory contains notebooks that illustrate fundamental concepts and workflows in GAICo in various domains. These examples are designed for users who want to understand the GAICo's functionalities in various domains.

### 1. [`example-timeseries.ipynb`](basic-examples/example-timeseries.ipynb): Time Series Analysis

- Evaluation and perturbation of time series using GAICo's time series specific metrics.

### 2. [`example-planning.ipynb`](basic-examples/example-planning.ipynb): Automated Planning Analysis

- Evaluation of various travel plans using GAICo's planning specific metrics.

### 3. [`example-finance.ipynb`](basic-examples/example-finance.ipynb): Finance Dataset Analysis

- Evaluating models on various questions from the finance domain by iterating on the dataset with the `Experiment` class.

### 4. [`example-recipes.ipynb`](basic-examples/example-recipes.ipynb): Recipe Dataset Analysis

- Evaluating models on various questions from the recipe domain by iterating on the dataset with the `Experiment` class. Further uses parallelization of the comparisons using `joblib`.

### 5. [`example-election.ipynb`](basic-examples/example-election.ipynb): Election Dataset Analysis

- Evaluating models on various questions from the election domain by using the `calculate()` metric method.

### 6. [`DeepSeek-example.ipynb`](basic-examples/example-DeepSeek.ipynb): Testing _DeepSeek R1_

- The aim for this notebook was to aid with evaluating DeepSeek R1 for [AI4Society's Point of View (POV)](https://drive.google.com/file/d/1ErR1xT7ftvmHiUyYrdUbjyd4qCK_FxKX/view?usp=sharing).
- **Note**: All results remove the `<think>` tags for the DeepSeek models.

## Advanced Examples

The `advanced-examples` directory contains advances notebooks showcasing more complex use cases and metrics. These examples are intended for users who are already familiar with the basics of GAICo. Please refer to the README.md file in that directory for details. A quick description:

### 1. [`example-llm_faq.ipynb`](advanced-examples/example-llm_faq.ipynb): LLM FAQ Analysis

- Comparison of various LLM responses (Phi, Mixtral, etc.) on FAQ dataset from USC.

### 2. [`example-threshold.ipynb`](advanced-examples/example-threshold.ipynb)`: Thresholds

- Exploration of default and custom thresholding techniques for LLM responses.

### 3. [`example-viz.ipynb`](advanced-examples/example-viz.ipynb): Visualizations

- Hands-on visualizations for LLM results.

## Citations

- `example-1.ipynb` and `example-2.ipynb`

  ```
  Srivastava, B., Lakkaraju, K., Gupta, N., Nagpal, V., Muppasani, B. C., & Jones, S. E. (2025). SafeChat: A Framework for Building Trustworthy Collaborative Assistants and a Case Study of its Usefulness. arXiv preprint arXiv:2504.07995.
  ```

- `example-planning.ipynb`

  ```
  Pallagani, V., Gupta, N., Aydin, J., & Srivastava, B. (2025). FABLE: A Novel Data-Flow Analysis Benchmark on Procedural Text for Large Language Model Evaluation. arXiv preprint arXiv:2505.24258.
  ```

- `example-finance.ipynb`

  ```
  Lakkaraju, K., Jones, S. E., Vuruma, S. K. R., Pallagani, V., Muppasani, B. C., & Srivastava, B. (2023, November). Llms for financial advisement: A fairness and efficacy study in personal decision making. In Proceedings of the Fourth ACM International Conference on AI in Finance (pp. 100-107).
  ```

- `example-recipe.ipynb`

  ```
  Nagpal, Vansh, et al. "A Novel Approach to Balance Convenience and Nutrition in Meals With Long-Term Group Recommendations and Reasoning on Multimodal Recipes and its Implementation in BEACON." arXiv preprint arXiv:2412.17910 (2024).
  ```

- `example-election.ipynb`

  ```
  Muppasani, B., Pallagani, V., Lakkaraju, K., Lei, S., Srivastava, B., Robertson, B., Hickerson, A. and Narayanan, V., 2023. On safe and usable chatbots for promoting voter participation. AI Magazine, 44(3), pp.240-247.
  ```
