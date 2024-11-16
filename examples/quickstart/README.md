# Quickstart

This is a quickstart guide to get you up and running with the `GenAIResultsComparator` library.

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
