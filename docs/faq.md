# Frequently Asked Questions (FAQ)

Here are answers to some common questions about using GAICo.

## Q: My Jupyter Notebook isn't working correctly with GAICo (e.g., import errors, version conflicts). What should I do?

This usually happens if Jupyter Notebook/Lab is not using the Python environment where GAICo was installed. Here's how to troubleshoot:

1.  **Ensure Virtual Environment Usage:**

    - Always create and activate a Python virtual environment before installing GAICo and Jupyter.
      ```shell
      # Example:
      python3 -m venv my-gaico-env
      source my-gaico-env/bin/activate  # macOS/Linux
      # my-gaico-env\Scripts\activate    # Windows
      ```
    - Install GAICo and Jupyter (Notebook/Lab) _into this activated environment_:
      ```shell
      pip install gaico notebook  # Or jupyterlab
      ```
    - Launch Jupyter from the same terminal where the virtual environment is active:
      ```shell
      jupyter notebook # Or jupyter lab
      ```

2.  **Select the Correct Kernel in Jupyter:**

    - When you open or create a notebook, Jupyter needs to use the kernel associated with your virtual environment.
    - In the Jupyter Notebook menu, go to **Kernel > Change kernel**.
    - You should see an option corresponding to your virtual environment. It might be named after the environment folder (e.g., "my-gaico-env"), simply "Python 3", or a display name you set up. Select it.
    - If the kernel restarts, allow it. Then try running your GAICo code again.

3.  **Explicitly Registering the Kernel (if the above doesn't resolve it):**
    If Jupyter consistently fails to find or use your virtual environment's kernel, you can make it explicitly available:

    - Make sure your virtual environment is active.
    - Install `ipykernel` if it's not already there (it's usually a dependency of `notebook` or `jupyterlab`):
      ```shell
      pip install ipykernel
      ```
    - Register the environment as a Jupyter kernel:
      ```shell
      python -m ipykernel install --user --name=my-gaico-env --display-name="Python (my-gaico-env)"
      ```
      (Replace `my-gaico-env` with your actual environment name and choose a descriptive `display-name`.)
    - Restart your Jupyter server (close the terminal window where it's running and relaunch it from the activated venv).
    - Now, in your notebook, go to **Kernel > Change kernel** and select the "Python (my-gaico-env)" (or your chosen display name).

4.  **Check `sys.path` and `sys.executable`:**
    If problems persist, add a cell to your notebook with the following to verify which Python environment is being used:

    ```python
    import sys
    import numpy # A core dependency
    import gaico

    print("Python Executable:", sys.executable)
    print("\nSystem Path:")
    for p in sys.path:
        print(p)
    print("\nNumPy version:", numpy.__version__)
    print("NumPy location:", numpy.__file__)
    print("\nGAICo location:", gaico.__file__)
    ```

    The `sys.executable` and the paths for `numpy` and `gaico` should all point to directories within your active virtual environment (e.g., `.../my-gaico-env/lib/python3.x/site-packages/...`). If they point to a global Python installation or a different environment, the kernel is not set up correctly.

## Q: How do I add a new custom metric?

Adding a new metric to GAICo is designed to be straightforward. All metrics inherit from the `BaseMetric` class.

1.  **Create a New Metric Class:**

    - Your new metric class should inherit from `gaico.metrics.base.BaseMetric`.
    - You must implement two methods:
      - `_single_calculate(self, generated_text: str, reference_text: str, **kwargs: Any) -> Union[float, dict]`: This method calculates the metric for a single pair of texts.
      - `_batch_calculate(self, generated_texts: Union[Iterable, np.ndarray, pd.Series], reference_texts: Union[Iterable, np.ndarray, pd.Series], **kwargs: Any) -> Union[List[float], List[dict], np.ndarray, pd.Series, float, dict]`: This method calculates the metric for a batch of texts. Often, this can be implemented by iterating and calling `_single_calculate`, but you can optimize it for batch operations if possible.
    - The public `calculate()` method is inherited from `BaseMetric` and handles input type checking and dispatching to either `_single_calculate` or `_batch_calculate`.

2.  **Example Structure:**

    ```python
    from typing import Any, Iterable, List, Union
    import numpy as np
    import pandas as pd
    from gaico.metrics.base import BaseMetric

    class MyCustomMetric(BaseMetric):
        def __init__(self, custom_param: str = "default"):
            self.custom_param = custom_param
            # Add any other initialization logic

        def _single_calculate(
            self, generated_text: str, reference_text: str, **kwargs: Any
        ) -> float:
            # Your logic to compare generated_text and reference_text
            # Example:
            score = 0.0
            if self.custom_param in generated_text and self.custom_param in reference_text:
                score = 1.0
            # Ensure score is normalized between 0 and 1
            return score

        def _batch_calculate(
            self,
            generated_texts: Union[Iterable, np.ndarray, pd.Series],
            reference_texts: Union[Iterable, np.ndarray, pd.Series],
            **kwargs: Any,
        ) -> Union[List[float], np.ndarray, pd.Series]:
            # Default batch implementation (can be optimized)
            results = [
                self._single_calculate(gen, ref, **kwargs)
                for gen, ref in zip(generated_texts, reference_texts)
            ]
            if isinstance(generated_texts, np.ndarray):
                return np.array(results)
            elif isinstance(generated_texts, pd.Series):
                return pd.Series(results, index=generated_texts.index)
            return results
    ```

3.  **Using Your Custom Metric:**
    You can directly instantiate and use your custom metric:

    ```python
    from gaico.metrics.base import BaseMetric # Or your custom metric file
    # from your_module import MyCustomMetric # If defined in a separate file

    # Assuming MyCustomMetric is defined as above
    custom_metric = MyCustomMetric(custom_param="example")
    score = custom_metric.calculate("This is an example generated text.", "This is an example reference text.")
    print(f"Custom metric score: {score}")
    ```

4.  **(Optional) Registering with the `Experiment` Class:**
    If you want your custom metric to be usable by its string name within the `Experiment` class (e.g., `exp.compare(metrics=['MyCustomMetric'])`), you'll need to add it to the `REGISTERED_METRICS` dictionary in `gaico/experiment.py`:

    ```python
    # In gaico/experiment.py
    # ... other imports ...
    # from ..metrics.my_custom_metric_module import MyCustomMetric # Adjust import path

    REGISTERED_METRICS: Dict[str, type[BaseMetric]] = {
        "Jaccard": JaccardSimilarity,
        "Cosine": CosineSimilarity,
        # ... other default metrics ...
        "MyCustomMetric": MyCustomMetric, # Add your metric here
    }
    ```

    You would also typically add it to `DEFAULT_METRICS_TO_RUN` in the same file if you want it to run by default when `metrics=None` in `Experiment.compare()`.

    Refer to existing metric implementations in `gaico/metrics/` for more detailed examples (e.g., `text_similarity_metrics.py`).

## Q: How do I get the list of all supported metrics or use them directly?

There are a couple of ways to understand and access "supported metrics":

1.  **All metric classes available for direct instantiation from `gaico.metrics`:**
    These are all the metric classes defined in the `gaico.metrics` sub-package that are intended for public use. They are typically listed in `gaico.metrics.__init__.__all__`.

    - **To import and use specific metric classes by name:**
      This is the recommended way for clarity.

      ```python
      from gaico.metrics import JaccardSimilarity, BLEU, BERTScore

      jaccard_scorer = JaccardSimilarity()
      bleu_scorer = BLEU()
      bert_scorer = BERTScore(model_type="distilbert-base-uncased") # Example of passing params

      score1 = jaccard_scorer.calculate("text a", "text b")
      score2 = bleu_scorer.calculate("text a", "text b")
      ```

    - **To import all available metric classes using `*`:**
      This makes all metric classes from `gaico.metrics.__init__.__all__` available in your current namespace.

      ```python
      from gaico.metrics import *

      # Now you can use the class names directly
      jaccard_scorer = JaccardSimilarity()
      rouge_scorer = ROUGE()
      # ... and so on for all metrics in gaico.metrics.__all__

      score = jaccard_scorer.calculate("text a", "text b")
      ```

      While convenient, using `import *` can sometimes make it less clear where names are coming from in larger projects. For `gaico.metrics`, which is focused, it's generally acceptable.

    - **To see what's available (programmatically):**

      ```python
      import gaico.metrics as gaico_metrics

      # Metrics explicitly listed in gaico.metrics.__init__.__all__
      available_metric_class_names = gaico_metrics.__all__
      print("Available metric classes for direct use (class names):", available_metric_class_names)
      # Example Output:
      # Available metric classes for direct use (class names): ['BLEU', 'ROUGE', 'JSDivergence', 'BERTScore', 'CosineSimilarity', 'JaccardSimilarity', 'LevenshteinDistance', 'SequenceMatcherSimilarity']
      ```

    Note that the class names (e.g., `JaccardSimilarity`) might be different from the shorter keys used in `REGISTERED_METRICS` (e.g., `Jaccard`).

2.  **Metrics available by string name in the `Experiment` class:**
    These are the metrics registered in `gaico.experiment.REGISTERED_METRICS`. You can access this dictionary programmatically to see which short names are recognized by the `Experiment` class:

    ```python
    from gaico.experiment import REGISTERED_METRICS

    registered_metric_names = list(REGISTERED_METRICS.keys())
    print("Metrics usable by name in Experiment class:", registered_metric_names)
    # Example Output:
    # ['Jaccard', 'Cosine', 'Levenshtein', 'SequenceMatcher', 'BLEU', 'ROUGE', 'JSD', 'BERTScore']
    ```

    The `gaico.experiment.DEFAULT_METRICS_TO_RUN` list also shows which of these are run by default if no specific metrics are requested in `Experiment.compare()`:

    ```python
    from gaico.experiment import DEFAULT_METRICS_TO_RUN

    print("Default metrics for Experiment class:", DEFAULT_METRICS_TO_RUN)
    ```

---

_If you have other questions, please [open an issue](https://github.com/ai4society/GenAIResultsComparator/issues) on our GitHub repository!_
