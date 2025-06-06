# Using Metrics Directly

While the [Experiment Class](experiment_class.md) provides a high-level interface for common evaluation scenarios, GAICo also allows you to use its individual metric classes directly. This approach offers more granular control and flexibility, especially when:

*   You need to evaluate a single model's output against a reference.
*   You are comparing lists of generated texts against corresponding lists of reference texts (i.e., pair-wise comparisons for multiple examples).
*   You want to integrate a specific metric into a custom evaluation pipeline.
*   You need to configure metric-specific parameters not exposed by the `Experiment` class's default initialization.
*   You are developing or testing a new custom metric.

## The `BaseMetric` Class

All metric classes in GAICo (e.g., `JaccardSimilarity`, `ROUGE`, `BERTScore`) inherit from the `gaico.metrics.base.BaseMetric` abstract class. This base class defines the common interface for all metrics, primarily through the `calculate()` method.

## Core Method: `calculate()`

The `calculate()` method is the primary way to compute a metric's score. It's designed to be flexible and can handle:

*   **Single Pair of Texts:** Comparing one generated text string to one reference text string.
*   **Batch of Texts:** Comparing an iterable (list, NumPy array, Pandas Series) of generated texts to a corresponding iterable of reference texts.
*   **Broadcasting:** Comparing a single generated text to multiple reference texts, or multiple generated texts to a single reference text.

### Parameters of `calculate()`:

*   `generated_texts` (`str | Iterable | np.ndarray | pd.Series`):
    A single generated text string or an iterable of generated texts.
*   `reference_texts` (`str | Iterable | np.ndarray | pd.Series`):
    A single reference text string or an iterable of reference texts.
*   `**kwargs`: Additional keyword arguments specific to the metric being used (e.g., `use_corpus_bleu=False` for `BLEU`, or `output_val=['f1']` for `BERTScore`).

### Return Value of `calculate()`:

The return type depends on the metric and the input:
*   For most metrics, it returns a `float` for single inputs or a `List[float]` (or `np.ndarray`/`pd.Series` if inputs were such) for batch inputs.
*   Metrics like `ROUGE` or `BERTScore` can return a `dict` of scores (e.g., `{'rouge1': 0.8, 'rougeL': 0.75}`) for single inputs or a `List[dict]` for batch inputs, depending on their configuration.

## Examples

Let's look at how to use some of the individual metric classes.

### 1. Jaccard Similarity

```python
from gaico.metrics import JaccardSimilarity

# Initialize the metric
jaccard_metric = JaccardSimilarity()

#  Single Pair
generated_text_single = "The quick brown fox"
reference_text_single = "A quick brown dog"
score_single = jaccard_metric.calculate(generated_text_single, reference_text_single)
print(f"Jaccard Score (Single): {score_single}") # Output: Jaccard Score (Single): 0.3333333333333333

#  Batch of Texts
generated_texts_batch = ["Hello world", "GAICo is great"]
reference_texts_batch = ["Hello there world", "GAICo is an awesome library"]
scores_batch = jaccard_metric.calculate(generated_texts_batch, reference_texts_batch)
print(f"Jaccard Scores (Batch): {scores_batch}")
# Jaccard Scores (Batch): [0.6666666666666666, 0.3333333333333333]

#  Broadcasting: Single generated text to multiple references
generated_text_broadcast = "Common evaluation text"
reference_texts_list = ["Evaluation text for comparison", "Another reference text"]
scores_broadcast_gen = jaccard_metric.calculate(generated_text_broadcast, reference_texts_list)
print(f"Jaccard Scores (Broadcast Gen): {scores_broadcast_gen}")
# Jaccard Scores (Broadcast Gen): [0.4, 0.2]
```

### 2. ROUGE Score (with specific configuration)

The `ROUGE` metric, by default, calculates 'rouge1', 'rouge2', and 'rougeL' F1-scores. You can customize this.

```python
from gaico.metrics import ROUGE

# Initialize ROUGE to calculate only 'rouge1' and 'rougeL' F1-scores
rouge_metric = ROUGE(rouge_types=['rouge1', 'rougeL'], use_stemmer=True)

generated = "The cat sat on the mat."
reference = "A cat was sitting on a mat."

# Calculate ROUGE scores
rouge_scores = rouge_metric.calculate(generated, reference)
print(f"ROUGE Scores: {rouge_scores}")
# Example Output: ROUGE Scores: {'rouge1': 0.4615384615384615, 'rougeL': 0.4615384615384615}

# If you configure for a single ROUGE type, it returns a float
rouge_metric_single_type = ROUGE(rouge_types=['rougeL'])
rouge_l_score = rouge_metric_single_type.calculate(generated, reference)
print(f"ROUGE-L Score: {rouge_l_score}") # Example Output: ROUGE-L Score: 0.4615384615384615
```

### 3. BERTScore (with specific output)

`BERTScore` can also be configured to return specific components (precision, recall, or F1) or a dictionary of them.

```python
from gaico.metrics import BERTScore

# Initialize BERTScore to return only the F1 score
# Note: BERTScore can be slow to initialize the first time as it downloads models.
# For faster tests/examples, you might use a smaller model or mock it.

# To get a dictionary with only F1 scores:
bertscore_metric_f1_dict = BERTScore()
generated_bert = "This is a test sentence for BERTScore."
reference_bert = "This is a reference sentence for BERTScore evaluation."
bert_f1_dict = bertscore_metric_f1_dict.calculate(generated_bert, reference_bert)
print(f"BERTScore (F1 dict): {bert_f1_dict}")
# Example Output: BERTScore (F1 dict): {'precision': 0.9229249954223633, 'recall': 0.8905344009399414, 'f1': 0.9064403772354126}

bertscore_metric_f1 = BERTScore(output_val=['f1']) # Returns a dict: {'f1': value}
bert_f1_score_float = bertscore_metric_f1.calculate(generated_bert, reference_bert) # Using the same instance
print(f"BERTScore (F1 float): {bert_f1_score_float}") # This will be the float value of F1
# Example Output: BERTScore (F1 float): 0.9064403772354126...
```

## Available Metrics

GAICo includes the following built-in metrics, all usable directly:

*   **N-gram-based Metrics:**
    *   `gaico.metrics.BLEU`
    *   `gaico.metrics.ROUGE`
    *   `gaico.metrics.JSDivergence` (Jensen-Shannon Divergence)
*   **Text Similarity Metrics:**
    *   `gaico.metrics.JaccardSimilarity`
    *   `gaico.metrics.CosineSimilarity`
    *   `gaico.metrics.LevenshteinDistance`
    *   `gaico.metrics.SequenceMatcherSimilarity`
*   **Semantic Similarity Metrics:**
    *   `gaico.metrics.BERTScore`

Refer to the [API Reference](../api/metrics/index.md) for detailed constructor parameters and any specific `**kwargs` for each metric's `calculate()` method.

Using metrics directly provides the foundational building blocks for more complex evaluation setups or for when you need precise control over individual metric calculations.
