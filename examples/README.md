# GAICo Examples

This directory contains example notebooks demonstrating various use cases of the GAICo library. Each example is designed to showcase practical applications and provide detailed guidance on using different metrics for specific scenarios.

## Table of Contents

- [Examples](#examples)
- [Advanced Examples](#advanced-examples)
- [Citations](#citations)

## Examples

### 1. `quickstart.ipynb`

- Using GAICo's `Experiment` module to provide a simple, quickstart workflow.

### 2. `example-1.ipynb`: Multiple Models, Single Metric

- Evaluating multiple models (LLMs, Google, and Custom) using a single metric with `<metric>.calculate()` method.

### 3. `example-2.ipynb`: Single Model, Multiple Metric

- Evaluating a single model on multiple metrics with their `<metric>.calculate()` methods.

### 4. `example-finance.ipynb`: Finance Dataset Analysis

- Evaluating models on various questions from the finance domain by iterating on the dataset with the `Experiment` class.

### 5. `example-recipe.ipynb`: Recipe Dataset Analysis

- Evaluating models on various questions from the recipe domain by iterating on the dataset with the `Experiment` class. Further uses parallelization of the comparisons using `joblib`.

### 6. `example-election.ipynb`: Election Dataset Analysis

- Evaluating models on various questions from the election domain by using the `calculate()` metric method.

### 7. `DeepSeek-example.ipynb`: Testing _DeepSeek R1_

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

## Citations

- `example-1.ipynb` and `example-2.ipynb`

  ```
  Srivastava, B., Lakkaraju, K., Gupta, N., Nagpal, V., Muppasani, B. C., & Jones, S. E. (2025). SafeChat: A Framework for Building Trustworthy Collaborative Assistants and a Case Study of its Usefulness. arXiv preprint arXiv:2504.07995.
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
