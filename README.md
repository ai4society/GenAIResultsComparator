# GenAIResultsComparator

1. Run the following commands to create a conda environment with the necessary packages: 
    conda create --name myenv --file requirements.txt 
    conda activate myenv
2. Add your dataset to the 'data/' directory or you can use the existing ones.
3. Run the following command to evaluate the LLM responses for your data: 'python code/main.py <metric_computation_function> <path_to_data_csv> <column_with_expected_responses> <column_with_generated_responses>'
4. Example usage: python 'code/main.py compute_jaccard data/sc_election_faqs_llama3.csv Expected Actual'