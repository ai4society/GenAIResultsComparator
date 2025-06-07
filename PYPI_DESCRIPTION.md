# GAICo: GenAI Results Comparator

**Repository:** [github.com/ai4society/GenAIResultsComparator](https://github.com/ai4society/GenAIResultsComparator)

**Documentation:** [ai4society.github.io/projects/GenAIResultsComparator](https://ai4society.github.io/projects/GenAIResultsComparator/index.html)

## Overview

GenAI Results Comparator, GAICo, is a Python library designed to help compare, analyze, and visualize outputs from Large Language Models (LLMs), often against a reference text. It provides a range of extensible metrics from the literature to facilitate comprehensive evaluation.

At the core, the library provides a set of metrics for evaluating text strings given as inputs and produce outputs on a scale of 0 to 1 (normalized), where 1 indicates a perfect match between the two texts. These scores are use to analyze LLM outputs as well as visualize.

## Quickstart

GAICo's `Experiment` class offers a streamlined workflow for comparing multiple model outputs, applying thresholds, generating plots, and creating CSV reports.

Here's a quick example:

```python
from gaico import Experiment

# Sample data
llm_responses = {
    "Model A": "Title: Jimmy Kimmel Reacts to Donald Trump Winning the Presidential ... Snippet: Nov 6, 2024 ...",
    "Model B": "I'm an AI and I don't have the ability to predict the outcome of elections.",
    "Model C": "Sorry, I am designed not to answer such a question.",
}
reference_answer = "Sorry, I am unable to answer such a question as it is not appropriate."

# 1. Initialize Experiment
exp = Experiment(
    llm_responses=llm_responses,
    reference_answer=reference_answer
)

# 2. Compare models using specific metrics
# This calculates scores, generates a plot (if plot=True), and saves a CSV report.
results_df = exp.compare(
    metrics=['Jaccard', 'ROUGE'],  # Specify metrics, or None for all defaults
    plot=True,                     # Set to True to display plots
    output_csv_path="experiment_report.csv",
    custom_thresholds={"Jaccard": 0.6, "ROUGE_rouge1": 0.35} # Optional
)

# The returned DataFrame contains the calculated scores
print("Scores DataFrame from compare():")
print(results_df)
```

For more detailed examples, please refer to our Jupyter Notebooks in the [`examples/`](https://github.com/ai4society/GenAIResultsComparator/tree/main/examples) folder in the repository.

## Features

- Implements various metrics for text comparison:
  - N-gram-based metrics (_BLEU_, _ROUGE_, _JS divergence_)
  - Text similarity metrics (_Jaccard_, _Cosine_, _Levenshtein_, _Sequence Matcher_)
  - Semantic similarity metrics (_BERTScore_)
- Provides visualization capabilities using matplotlib and seaborn for plots like bar charts and radar plots.
- Allows exportation of results to CSV files, including scores and threshold pass/fail status.
- Provides streamlined `Experiment` class for easy comparison of multiple models, applying thresholds, plotting, and reporting.
- Supports batch processing for efficient computation.
- Optimized for different input types (lists, numpy arrays, pandas Series).
- Has extendable architecture for easy addition of new metrics.
- Supports automated testing of metrics using [Pytest](https://docs.pytest.org/en/stable/).

## Installation

You can install GAICo directly from PyPI using pip:

```shell
pip install gaico
```

The default installation includes core metrics. For optional features, you can install extras:

- To include the **BERTScore** metric (which has larger dependencies like PyTorch):
  ```shell
  pip install gaico[bertscore]
  ```
- To include the **JSDivergence** metric (requires SciPy):
  ```shell
  pip install gaico[jsd]
  ```
- To install with **all optional features** mentioned above:
  ```shell
  pip install gaco[bertscore,cosine,jsd]
  ```

### Installation Size Comparison

The following table provides an _estimated_ overview of the relative disk space impact of different installation options. Actual sizes may vary depending on your operating system, Python version, and existing packages. These are primarily to illustrate the relative impact of optional dependencies.

_Note:_ Core dependencies include: `levenshtein`, `matplotlib`, `numpy`, `pandas`, `rouge-score`, and `seaborn`.

| Installation Command                      | Dependencies                                                 | Estimated Total Size Impact |
| ----------------------------------------- | ------------------------------------------------------------ | --------------------------- |
| `pip install gaico`                       | Core                                                         | 210 MB                      |
| `pip install gaico[jsd]`                  | Core + `scipy`                                               | 310 MB                      |
| `pip install gaico[cosine]`               | Core + `scikit-learn`                                        | 360 MB                      |
| `pip install gaico[bertscore]`            | Core + `bert-score` (includes `torch`, `transformers`, etc.) | 800 MB                      |
| `pip install gaico[bertscore,cosine,jsd]` | Core + all dependencies from above                           | 950 MB                      |

## Citation

If you find GAICo useful in your research or work, please consider citing it:

```bibtex
@software{AI4Society_GAICo_GenAI_Results,
  author = {{Nitin Gupta, Pallav Koppisetti, Biplav Srivastava}},
  license = {MIT},
  title = {{GAICo: GenAI Results Comparator}},
  year = {2025},
  url = {https://github.com/ai4society/GenAIResultsComparator}
}
```
