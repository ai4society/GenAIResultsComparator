# GAICo: GenAI Results Comparator

<!-- BADGES_START -->
<p align="center">
  <a href="https://pypi.org/project/GAICo/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/GAICo.svg?style=flat-square"></a>
  <a href="https://pypi.org/project/GAICo/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/GAICo.svg?style=flat-square"></a>
  <a href="https://github.com/ai4society/GenAIResultsComparator/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/ai4society/GenAIResultsComparator?style=flat-square"></a>
  <a href="https://ai4society.github.io/projects/GenAIResultsComparator/"><img alt="Documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat-square"></a>
  <a href="https://github.com/ai4society/GenAIResultsComparator/actions/workflows/deploy-docs.yml"><img alt="Deploy Docs" src="https://github.com/ai4society/GenAIResultsComparator/actions/workflows/deploy-docs.yml/badge.svg?branch=main&style=flat-square"></a>

  <!-- Uncomment below if repository blows up -->
  <!-- <br/>
  <a href="https://pypistats.org/packages/gaico"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/GAICo.svg?style=flat-square"></a>
  <a href="https://github.com/ai4society/GenAIResultsComparator/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/ai4society/GenAIResultsComparator?style=social"></a>
  <a href="https://github.com/ai4society/GenAIResultsComparator/network/members"><img alt="GitHub Forks" src="https://img.shields.io/github/forks/ai4society/GenAIResultsComparator?style=social"></a> -->
</p>
<!-- BADGES_END -->

<!-- TAGLINE_START -->

_GenAI Results Comparator, GAICo, is a Python library_ to help compare, analyze and visualize outputs from Large Language Models (LLMs), often against a reference text. In doing so, one can use a range of extensible metrics from the literature.

<!-- TAGLINE_END -->

Important Links:

- Documentation: [ai4society.github.io/projects/GenAIResultsComparator](https://ai4society.github.io/projects/GenAIResultsComparator).

- FAQ: [ai4society.github.io/projects/GenAIResultsComparator/faq](https://ai4society.github.io/projects/GenAIResultsComparator/faq).

- PyPI: [pypi.org/project/gaico/](https://pypi.org/project/gaico/)

## Quick Start

GAICo makes it easy to evaluate and compare LLM outputs. For detailed, runnable examples, please refer to our Jupyter Notebooks in the [`examples/`](examples/) folder:

- [`quickstart.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb): Rapid hands-on with the _Experiment_ sub-module.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/quickstart.ipynb)

- [`example-1.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb): For fine-grained usage, this notebook focuses on comparing **multiple model outputs** using a **single metric**.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-1.ipynb)

- [`example-2.ipynb`](https://github.com/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb): For fine-grained usage, this notebook demonstrates evaluating a **single model output** across **all available metrics**.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4society/GenAIResultsComparator/blob/main/examples/example-2.ipynb)

## Streamlined Workflow with _`Experiment`_

For a more integrated approach to comparing multiple models, applying thresholds, generating plots, and creating CSV reports, the `Experiment` class offers a convenient abstraction.

### Quick Example

This example demonstrates comparing multiple LLM responses against a reference answer using specified metrics, generating a plot, and outputting a CSV report.

<!-- QUICKSTART_CODE_START -->

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
#   This will calculate scores for 'Jaccard' and 'ROUGE',
#   generate a plot (e.g., radar plot for multiple metrics/models),
#   and save a CSV report.
results_df = exp.compare(
    metrics=['Jaccard', 'ROUGE'],  # Specify metrics, or None for all defaults
    plot=True,
    output_csv_path="experiment_report.csv",
    custom_thresholds={"Jaccard": 0.6, "ROUGE_rouge1": 0.35} # Optional: override default thresholds
)

# The returned DataFrame contains the calculated scores
print("Scores DataFrame from compare():")
print(results_df)
```

<!-- QUICKSTART_CODE_END -->

This abstraction streamlines common evaluation tasks, while still allowing access to the underlying metric classes and dataframes for more advanced or customized use cases. More details in [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

### Scope and Dataset Evaluation

The `Experiment` class is designed for evaluating a set of model responses against a **single reference answer** at a time. This is ideal for analyzing outputs for a specific prompt or scenario.

Alternatively, if you have a dataset consisting of multiple reference texts and corresponding sets of generated texts from various models (e.g., `list_of_references`, `list_of_model_A_generations`, etc.), you can use the individual metric classes (e.g., `JaccardSimilarity().calculate(list_of_gens, list_of_refs)`), which support list inputs. This approach offers more control but requires more manual orchestration of results.

If you would like to use the `Experiment` abstraction for your dataset, you would need to iterate through your dataset and create an `Experiment` instance for each data point (i.e., for each reference text and its associated model responses). Examples of such scenarios are shown in the [`examples`](examples) subdirectory, please refer to the README.md file there.

Lastly, for you implementing custom metrics, please look at the [FAQs](https://ai4society.github.io/projects/GenAIResultsComparator/faq/#q-how-do-i-add-a-new-custom-metric).

<p align="center">
  <img src="https://raw.githubusercontent.com/ai4society/GenAIResultsComparator/refs/heads/main/examples/data/examples/example_2.png" alt="Sample Radar Chart showing multiple metrics for a single LLM" width="450"/>
  <br/><em>Example Radar Chart generated by the <code>examples/example-2.ipynb</code> notebook.</em>
</p>

<!-- DESCRIPTION_FULL_START -->

## Description

<!-- DESCRIPTION_CORE_CONCEPT_START -->

At the core, the library provides a set of metrics for evaluating text strings given as inputs and produce outputs on a scale of 0 to 1 (normalized), where 1 indicates a perfect match between the texts. These scores are use to analyze LLM outputs as well as visualize.

<!-- DESCRIPTION_CORE_CONCEPT_END -->

**_Class Structure:_** All metrics are implemented as classes, and they can be easily extended to add new metrics. The metrics start with the `BaseMetric` class under the `gaico/base.py` file.

Each metric class inherits from this base class and is implemented with **just one required method**: `calculate()`.

This `calculate()` method takes two parameters:

- `generated_texts`: Either a string or a Iterables of strings representing the texts generated by an LLM.
- `reference_texts`: Either a string or a Iterables of strings representing the expected or reference texts.

If the inputs are Iterables (lists, Numpy arrays, etc.), then the method assumes that there exists a one-to-one mapping between the generated texts and reference texts, meaning that the first generated text corresponds to the first reference text, and so on.

**_Notes:_**

- While the library can be used to compare strings, it's main purpose is to aid with comparing results from various LLMs.
- Due to size constraints, `pip install gaico` will install 5/8 [metrics](#features) supported by the library. For the remaining, the library supports [optional installs](#optional-installations-for-gaico), which would only add the needed metric.

**_Inspiration_** for the library and evaluation metrics was taken from [Microsoft's
article on evaluating LLM-generated content](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/list-of-eval-metrics). In the article, Microsoft describes 3 categories of evaluation metrics: **(1)** Reference-based metrics, **(2)** Reference-free metrics, and **(3)** LLM-based metrics. _The library currently supports reference-based metrics._

<!-- DESCRIPTION_FULL_END -->

<p align="center">
  <img src="https://raw.githubusercontent.com/ai4society/GenAIResultsComparator/refs/heads/main/gaico.drawio.png" alt="GAICo Overview">
</p>
<p align="center">
  <em>Overview of the workflow supported by the <i>GAICo</i> library</em>
</p>

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

<!-- FEATURES_SECTION_START -->

## Features

<!-- FEATURES_LIST_START -->

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
  <!-- FEATURES_LIST_END -->
  <!-- FEATURES_SECTION_END -->

<!-- INSTALLATION_SECTION_START -->

## Installation

<!-- INSTALLATION_STANDARD_INTRO_START -->

GAICo can be installed using pip. We strongly recommend using a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) to manage dependencies and avoid conflicts with other packages.

<!-- INSTALLATION_STANDARD_INTRO_END -->

<!-- INSTALLATION_STANDARD_SETUP_START -->

- **Create and activate a virtual environment** (e.g., named `gaico-env`):

  ```shell
    # For Python 3.10+
    python3 -m venv gaico-env
    source gaico-env/bin/activate  # On macOS/Linux
    # gaico-env\Scripts\activate   # On Windows
  ```

<!-- INSTALLATION_STANDARD_SETUP_END -->
<!-- INSTALLATION_PYPI_BASIC_START -->

- **Install GAICo:**
  Once your virtual environment is active, install GAICo using pip:

  ```shell
    pip install gaico
  ```

This installs the core GAICo library.

<!-- INSTALLATION_PYPI_BASIC_END -->

<!-- INSTALLATION_JUPYTER_GUIDE_START -->

### Using GAICo with Jupyter Notebooks/Lab

If you plan to use GAICo within Jupyter Notebooks or JupyterLab (recommended for exploring examples and interactive analysis), install them into the _same activated virtual environment_:

```shell
# (Ensure your 'gaico-env' is active)
pip install notebook  # For Jupyter Notebook
# OR
# pip install jupyterlab # For JupyterLab
```

Then, launch Jupyter from the same terminal where your virtual environment is active:

```shell
# (Ensure your 'gaico-env' is active)
jupyter notebook
# OR
# jupyter lab
```

New notebooks created in this session should automatically use the `gaico-env` Python environment. For troubleshooting kernel issues, please see our [FAQ document](https://ai4society.github.io/projects/GenAIResultsComparator/faq).

<!-- INSTALLATION_JUPYTER_GUIDE_END -->

<!-- INSTALLATION_OPTIONAL_INTRO_START -->

### Optional Installations for GAICo

The default installation includes core metrics and is lightweight. For optional features and metrics that have larger dependencies:

<!-- INSTALLATION_OPTIONAL_INTRO_END -->

<!-- INSTALLATION_OPTIONAL_FEATURES_START -->

- To include the **BERTScore** metric (which has larger dependencies like PyTorch):
  ```shell
  pip install 'gaico[bertscore]'
  ```
- To include the **CosineSimilarity** metric (requires scikit-learn):
  ```shell
  pip install 'gaico[cosine]'
  ```
- To include the **JSDivergence** metric (requires SciPy and NLTK):
  ```shell
  pip install 'gaico[jsd]'
  ```
- To install with **all optional features**:
  ```shell
  pip install 'gaico[bertscore,cosine,jsd]'
  ```
  _(Note: All optional features are also installed if you use the `dev` extra for development installs.)_
  <!-- INSTALLATION_OPTIONAL_FEATURES_END -->

<!-- INSTALLATION_SIZE_TABLE_INTRO_START -->

### Installation Size Comparison

<!-- INSTALLATION_SIZE_TABLE_INTRO_END -->

<!-- INSTALLATION_SIZE_TABLE_CONTENT_START -->

The following table provides an _estimated_ overview of the relative disk space impact of different installation options. Actual sizes may vary depending on your operating system, Python version, and existing packages. These are primarily to illustrate the relative impact of optional dependencies.

_Note:_ Core dependencies include: `levenshtein`, `matplotlib`, `numpy`, `pandas`, `rouge-score`, and `seaborn`.

| Installation Command                        | Dependencies                                                 | Estimated Total Size Impact |
| ------------------------------------------- | ------------------------------------------------------------ | --------------------------- |
| `pip install gaico`                         | Core                                                         | 210 MB                      |
| `pip install 'gaico[jsd]'`                  | Core + `scipy`, `nltk`                                       | 310 MB                      |
| `pip install 'gaico[cosine]'`               | Core + `scikit-learn`                                        | 360 MB                      |
| `pip install 'gaico[bertscore]'`            | Core + `bert-score` (includes `torch`, `transformers`, etc.) | 800 MB                      |
| `pip install 'gaico[bertscore,cosine,jsd]'` | Core + all dependencies from above                           | 950 MB                      |

<!-- INSTALLATION_SIZE_TABLE_CONTENT_END -->

<!-- INSTALLATION_DEVELOPER_GUIDE_START -->

### For Developers (Installing from source)

If you want to contribute to GAICo or install it from source for development:

1.  Clone the repository:

    ```shell
    git clone https://github.com/ai4society/GenAIResultsComparator.git
    cd GenAIResultsComparator
    ```

2.  Set up a virtual environment and install dependencies:
    We recommend using [UV](https://docs.astral.sh/uv/#installation) for managing environments and dependencies.

    ```shell
    # Create a virtual environment (e.g., Python 3.10-3.12 recommended)
    uv venv
    # Activate the environment
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    # Install the package in editable mode with all development dependencies
    # (includes all optional features like bertscore, cosine, jsd)
    uv pip install -e ".[dev]"
    ```

    _If you prefer not to use `uv`,_ you can use `pip`:

    ```shell
    # Create a virtual environment (e.g., Python 3.10-3.12 recommended)
    python3 -m venv .venv
    # Activate the environment
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    # Install the package in editable mode with development extras
    pip install -e ".[dev]"
    ```

    _(The `dev` extra installs GAICo with all its optional features, plus dependencies for testing, linting, building, and documentation.)_

3.  Set up pre-commit hooks (optional but recommended for contributors):

    ```shell
    pre-commit install
    ```

    <!-- INSTALLATION_DEVELOPER_GUIDE_END -->
    <!-- INSTALLATION_SECTION_END -->

## Project Structure

The project structure is as follows:

```shell
.
├── README.md
├── LICENSE
├── .gitignore
├── uv.lock
├── pyproject.toml
├── project_macros.py        # Used by mkdocs-macros-plugin (documentation)
├── PYPI_DESCRIPTION.MD      # The PyPI description file
├── .pre-commit-config.yaml  # Pre-Commit Hooks
├── .mkdocs.yml              # Configuration for mkdocs (documentation)
├── gaico/                   # Contains the library code
├── examples/                # Contains example scripts
├── tests/                   # Contains test
├── scripts/                 # Contains scripts for github deployment and markdown generation
├── docs/                    # Contains documentation files
└── .github/workflows/       # Contains workflows for deploying to PyPI and the documentations site.

```

### Code Style

We use `pre-commit` hooks to maintain code quality and consistency. The configuration for these hooks is in the `.pre-commit-config.yaml` file. These hooks run automatically on `git commit`, but you can also run them manually:

```
pre-commit run --all-files
```

## Running Tests

Navigate to the project root in your terminal and run:

```bash
uv run pytest
```

Or, for more verbose output:

```bash
uv run pytest -v
```

To skip the slow BERTScore tests:

```bash
uv run pytest -m "not bertscore"
```

To run only the slow BERTScore tests:

```bash
uv run pytest -m bertscore
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/FeatureName`)
3. Commit your changes (`git commit -m 'Add some FeatureName'`)
4. Push to the branch (`git push origin feature/FeatureName`)
5. Open a Pull Request

Please ensure that your code passes all tests and adheres to our code style guidelines (enforced by pre-commit hooks) before submitting a pull request.

<!-- CITATION_SECTION_START -->

## Citation

<!-- CITATION_CONTENT_START -->

If you find this project useful, please consider citing it in your work:

```bibtex
@software{AI4Society_GAICo_GenAI_Results,
  author = {{Nitin Gupta, Pallav Koppisetti, Biplav Srivastava}},
  license = {MIT},
  title = {{GAICo: GenAI Results Comparator}},
  year = {2025},
  url = {https://github.com/ai4society/GenAIResultsComparator}
}
```

<!-- CITATION_CONTENT_END -->
<!-- CITATION_SECTION_END -->

<!-- ACKNOWLEDGMENTS_SECTION_START -->

## Acknowledgments

- The library is developed by [Nitin Gupta](https://github.com/g-nitin), [Pallav Koppisetti](https://github.com/pallavkoppisetti), and [Biplav Srivastava](https://github.com/biplav-s). Members of [AI4Society](https://ai4society.github.io) contributed to this tool as part of ongoing discussions. Major contributors are credited.
- This library uses several open-source packages including NLTK, scikit-learn, and others. Special thanks to the creators and maintainers of the implemented metrics.

<!-- ACKNOWLEDGMENTS_SECTION_END -->

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ai4society/GenAIResultsComparator/blob/main/LICENSE) file for details.

<!-- CONTACT_SECTION_START -->

## Contact

If you have any questions, feel free to reach out to us at [ai4societyteam@gmail.com](mailto:ai4societyteam@gmail.com).

<!-- CONTACT_SECTION_END -->
