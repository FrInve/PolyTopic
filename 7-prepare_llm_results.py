import pickle
import numpy as np
import pandas as pd
import re
import json


def main_new():

    df_name = 'dbpedia'
    general_path = f"saved/{df_name}_ctm"

    with open(f"{general_path}/{df_name}_output_list_new", "rb") as f:
        loaded_list = pickle.load(f)
    
    print(len(loaded_list))

    evaluation_results = []

    for raw_text in loaded_list:
        parsed_text = parse_with_regex(raw_text)
        evaluation_results.append(parsed_text)
    
    with open(f"{general_path}/{df_name}_evaluation_results_new.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    

    positives = [0, 0, 0]
    sum_scores = 0
    questions = ['Q1', 'Q2', 'Q3']

    # evaluation_scores = []
    for i in range(len(evaluation_results)):
        for j in range(len(questions)):
            if evaluation_results[i][questions[j]]==True:
                positives[j] += 1
        sum_scores += evaluation_results[i]['Q4']

        
    print(f"Count for positives: {positives}")

    avg_score = float(sum_scores) / float(len(evaluation_results))

        #evaluation_scores.append([precision_score, recall_score, avg_score])

    # np.save(f"{general_path}/{df_name}_evaluation_scores.npy", evaluation_scores)

    print(f"Average score: {avg_score}")






def parse_with_regex(text):
    try:
        # Match everything from the first '{' to the last '}' inclusive
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON structure found in text.")
    except Exception as e:
        print(f"Error parsing: {e}")
        return None


if __name__ == "__main__":
    main_new()
