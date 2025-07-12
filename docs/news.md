# GAICo Release News

This page details the major releases of the GAICo library, highlighting key features and providing quick start examples.

## v0.2.0 - July 2025

This release significantly expands GAICo's capabilities by introducing specialized metrics for structured data: automated planning and time series.

### Key Features:

- **Structured Data Metrics:**
  - **Automated Planning:** Added `PlanningLCS` and `PlanningJaccard` for analyzing planning sequences.
  - **Time-Series:** Introduced metrics like `TimeSeriesElementDiff` and `TimeSeriesDTW` for evaluating time-series data.

### Quick Start Example:

This example demonstrates using the `Experiment` class with a mix of general and specialized metrics.

```python
from gaico import Experiment

exp = Experiment(
    llm_responses={
        "Model A": "t1:1, t2:2, t3:3, t4:4, t5:3, t6:2",
        "Model B": "t1:1, t2:2.1, t3:3.4, t4:8, t5:5",
    },
    reference_answer="t1:1, t2:2.2, t3:3.1, t4:4, t5:3.5",
)

# Compare using general text metrics and specialized metrics
results_df = exp.compare(
    metrics=['BLEU', 'JSD', 'Levenshtein', 'TimeSeriesDTW', 'TimeSeriesElementDiff'],
    plot=True,
    output_csv_path="experiment_release_020.csv"
)
```

<figure markdown="span">
  ![Quick Start Example Output](https://raw.githubusercontent.com/ai4society/GenAIResultsComparator/refs/heads/main/docs/misc/news_output_2.png){ width="600" }
  <figcaption><em>GAICo v0.2.0 Quick Start Example Output</em></figcaption>
</figure>

---

## v0.1.5 - June 2025

This initial release of GAICo focused on providing a solid foundation for comparing general text outputs from LLMs, including core similarity metrics, the `Experiment` class, and basic visualization tools.

### Key Features:

- **Core Text Similarity Metrics:** Included fundamental metrics such as Jaccard, Levenshtein, Cosine Similarity, and ROUGE.
- **`Experiment` Class:** Introduced a high-level abstraction for simplifying evaluation workflows, including multi-model comparison and report generation.
- **Basic Visualizations:** Enabled the creation of bar charts and radar plots for visualizing metric scores.
- **Extensible Architecture:** Designed for easy addition of new metrics.

### Quick Start Example:

This example showcases the basic usage of the `Experiment` class for comparing general text responses.

```python
from gaico import Experiment

# Sample data from https://arxiv.org/abs/2504.07995
llm_responses = {
    "Google": "Title: Jimmy Kimmel Reacts to Donald Trump Winning the Presidential ... Snippet: Nov 6, 2024 ...",
    "Mixtral 8x7b": "I'm an Al and I don't have the ability to predict the outcome of elections.",
    "SafeChat": "Sorry, I am designed not to answer such a question.",
}
reference_answer = "Sorry, I am unable to answer such a question as it is not appropriate."

# 1. Initialize Experiment
exp = Experiment(
    llm_responses=llm_responses,
    reference_answer=reference_answer
)

# 2. Compare models using specific metrics
results_df = exp.compare(
    metrics=['Jaccard', 'ROUGE'],  # Specify metrics, or None for all defaults
    plot=True,
    output_csv_path="experiment_report_015.csv"
)
```

<figure markdown="span">
  ![Quick Start Example Output](https://raw.githubusercontent.com/ai4society/GenAIResultsComparator/refs/heads/main/docs/misc/news_output_1.png){ width="600" }
  <figcaption><em>GAICo v0.1.5 Quick Start Example Output</em></figcaption>
</figure>
