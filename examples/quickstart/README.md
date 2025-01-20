# Quickstart

This guide contains quickstart examples to get you up and running with the `GenAIResultsComparator` library.
It also contains a [detailed usage](#detailed-usage) section for various metrics available in the library.

Here's a simple example of how to use GenAIResultsComparator.
_Please refer to the `quickstart_example.ipynb` file for a notebook-based version._

```python
from llm_metrics import BLEU, ROUGE, BERTScore

# Initialize metrics
bleu = BLEU()
rouge = ROUGE()
bert_score = BERTScore()

# Example texts
sentence_1 = "The quick brown fox jumps over the lazy dog"
sentence_2 = "A fast brown fox leaps over a sleepy canine"

# Calculate scores
bleu_score = bleu.calculate(sentence_1, sentence_2)
rouge_score = rouge.calculate(sentence_1, sentence_2)
bert_score = bert_score.calculate(sentence_1, sentence_2)

print(f"BLEU score: {bleu_score}")
print(f"ROUGE scores: {rouge_score}")
print(f"BERTScore: {bert_score}")

# For batch processing
generated_texts = [
    "The quick brown fox jumps over the lazy dog",
    "The cat chases the mouse"
]
reference_texts = [
    "A fast brown fox leaps over a sleepy canine",
    "A feline pursues a rodent"
]

# Batch calculate scores
bleu_scores = bleu.batch_calculate(generated_texts, reference_texts)
rouge_scores = rouge.batch_calculate(generated_texts, reference_texts)
bert_scores = bert_score.batch_calculate(generated_texts, reference_texts)
```

Furthermore, each metric can be customized during initialization:

```python
# Customize BLEU
bleu = BLEU(n=3)  # Use 3-grams instead of default 4-grams

# Customize ROUGE
rouge = ROUGE(rouge_types=['rouge1', 'rouge2'], use_stemmer=True)

# Customize BERTScore
bert_score = BERTScore(model_type='bert-base-uncased', num_layers=8)
```

Lastly, for advanced users, all metrics support additional parameters:

```python
# Pass additional parameters during calculation
bleu_score = bleu.calculate(text1, text2, additional_params={
    'smoothing_function': custom_smoothing
})
```


## Detailed Usage

### N-gram-based Metrics

* BLEU (Bilingual Evaluation Understudy)

    ```python
    from llm_metrics import BLEU

    bleu = BLEU(n=4)  # n is the maximum n-gram order
    score = bleu.calculate(generated_text, reference_text)
    ```

* ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

    ```python
    from llm_metrics import ROUGE

    rouge = ROUGE(rouge_types=['rouge1', 'rouge2', 'rougeL'])
    scores = rouge.calculate(generated_text, reference_text)
    ```

* JS Divergence

    ```python
    from llm_metrics import JSDivergence

    js_div = JSDivergence()
    score = js_div.calculate(generated_text, reference_text)
    ```

### Text Similarity Metrics

* Jaccard Similarity

    ```python
    from llm_metrics import JaccardSimilarity

    jaccard = JaccardSimilarity()
    score = jaccard.calculate(generated_text, reference_text)
    ```

* Cosine Similarity

    ```python
    from llm_metrics import CosineSimilarity

    cosine = CosineSimilarity()
    score = cosine.calculate(generated_text, reference_text)
    ```

* Levenshtein Distance

    ```python
    from llm_metrics import LevenshteinDistance

    levenshtein = LevenshteinDistance()
    distance = levenshtein.calculate(generated_text, reference_text)
    ```

### Semantic Similarity Metrics

* BERTScore

    ```python
    from llm_metrics import BERTScore

    bert_score = BERTScore(model_type="bert-base-uncased")
    score = bert_score.calculate(generated_text, reference_text)
    ```

### Batch Processing

All metrics support batch processing for efficient computation on multiple texts:

    ```python
    generated_texts = ["Text 1", "Text 2", "Text 3"]
    reference_texts = ["Ref 1", "Ref 2", "Ref 3"]

    scores = metric.batch_calculate(generated_texts, reference_texts)
    ```
