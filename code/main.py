# Import libraries
import numpy as np
import pandas as pd
import textdistance
import argparse

class JaccardDistance:
    @staticmethod
    def jaccard_distance(str1, str2):
        # Calculate the Jaccard distance between two strings.
        set1 = set(str1.split())
        set2 = set(str2.split())
        # If Jaccard distance = 0, the two strings are identical. Otherwise, different.
        return 1 - textdistance.jaccard.similarity(set1, set2)

    @staticmethod
    def compute_jaccard(data_path, true_col, pred_col):
        # Compute the average Jaccard distance for a dataset.
        data = pd.read_csv(data_path)
        data["jaccard_dist"] = data.apply(lambda row: JaccardDistance.jaccard_distance(row[true_col], row[pred_col]), axis=1)
        return np.mean(data["jaccard_dist"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Jaccard distance on a dataset.")
    parser.add_argument("function", choices=["compute_jaccard"], help="The function to call.")
    parser.add_argument('data', help="Path to the CSV file containing the dataset.")
    parser.add_argument('true_col', help="Name of the column containing the true values.")
    parser.add_argument('pred_col', help="Name of the column containing the predicted values.")
    
    args = parser.parse_args()

    if args.function == "compute_jaccard":
        result = JaccardDistance.compute_jaccard(args.data, args.true_col, args.pred_col)
        print(f"Average Jaccard Distance: {result}")
    else:
        print("No such function exists!")



