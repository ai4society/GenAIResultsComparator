# Deepseek Testing

These are tests comparing DeepSeek's models to other LLMs.

The `data` directory contains the data used for the tests. It contains the following files:
- `llm_responses.csv`: 1st testing dataset
- `generic_qa.json` and `huggingface.json`: 2nd testing datasets

As such, two tests are performed:
- `R1_faq_example_1.ipynb` contains the code for the 1st test.
- `R1_faq_example_2.ipynb` contains the code for the 2nd test.

The outputs for these tests, the metric scores (in pickle format) and the visualization plots are stored in the `results` and `plots` directories, respectively. Furthermore, the processed data files are saved under the `data/processed` directory.

**Note**: All results remove the `<think>` tags for the DeepSeek models.
