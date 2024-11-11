# GenAIResultsComparator Examples

This directory contains example notebooks demonstrating various use cases of the LLM-Metrics library. Each example is designed to showcase practical applications and provide detailed guidance on using different metrics for specific scenarios.

## Available Examples

### 1. Introduction to LLM-Metrics (`quickstart_example.ipynb`)
- Basic introduction to the library
- Detailed walkthrough of BLEU score implementation
- Best practices and common pitfalls
- Recommended for first-time users

### 2. LLM FAQ Analysis (`llm_faq/`)
- Comparison of responses from different LLMs on FAQ datasets
- Demonstrates how to:
  - Evaluate consistency across multiple LLM responses
  - Compare responses from different models
  - Analyze semantic similarity in FAQ contexts
- Uses multiple metrics including BLEU, BERTScore, and semantic similarity
- Includes sample FAQ dataset and responses

### 3. LLM-Aware Metrics (`llm_aware_metrics/`)
- Advanced metrics specifically designed for LLM output evaluation
- Demonstrates how to:
  - Compare LLM outputs while considering their prompts
  - Validate structured outputs against schemas
  - Measure prompt-response alignment
- Features three detailed example notebooks:
  - Basic prompt-aware comparison
  - Schema-based output validation
  - Alignment scoring
- Includes implementation code and comprehensive documentation
- Perfect for users working with LLM-generated content

## Contributing New Examples

We welcome contributions! If you'd like to add a new example:

1. Create a new directory for your example if it's more than a single notebook
2. Ensure your notebook is well-documented with:
   - Clear objective and use case
   - Requirements and setup instructions
   - Step-by-step explanations
   - Sample data (if applicable)
3. Submit a pull request with your example
