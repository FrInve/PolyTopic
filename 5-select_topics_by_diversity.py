import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import pickle


def main():

    # tart the mathematical stopwatch
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_datetime}] Execution started...")
    start_time = time.time()

    # Define Jaccard threshold
    jaccard_threshold = 0.3  # Adjust this based on how strict you want the filter to be
        

    # Define dataframe name and data path
    df_name = 'dbpedia'
    prob_path = f"saved/{df_name}_ctm"

    with open(f"{prob_path}/{df_name}_topics_dict.pkl", "rb") as f:
        topics_dict = pickle.load(f)
    print(f"Loaded topics dictionary")

    input_path = f"{prob_path}/{df_name}_relevant_topics.pkl"
    print(f"Loading selected topics from {input_path}...")

    with open(input_path, "rb") as f:
        relevant_topics_for_all = pickle.load(f)
        
    print(f"Loaded {len(relevant_topics_for_all)} documents. Applying MMR (Maximal Marginal Relevance)...")

    # Apply the diversity filter to every document
    final_diverse_topics_all = []
    
    for doc_idx, relevant_topics in enumerate(relevant_topics_for_all):
        diverse_topics = filter_topics_by_diversity(relevant_topics, jaccard_threshold, topics_dict)
        final_diverse_topics_all.append(diverse_topics)

    print(final_diverse_topics_all[0])

    # Count topics per document
    n_relevant_topics = []

    for i in range(len(final_diverse_topics_all)):
        n_relevant_topics.append(len(final_diverse_topics_all[i]))

    print("Max topics for document:", np.max(n_relevant_topics))
    print("Avg topics for document:", np.average(n_relevant_topics))


    count_relevant_topics = np.array([0]*11)

    for num in n_relevant_topics:
        count_relevant_topics[num] += 1

    print("Number of documents with n topics:", count_relevant_topics)

    count_doc_chars = np.array([0]*11)
    count_doc_words = np.array([0]*11)

    df_text = pd.read_parquet("data/processed/parquet/"+ df_name + "_text.parquet")
    docs = df_text['text'].to_list()

    for i in range(len(docs)):
        chars = len(docs[i])
        count_doc_chars[n_relevant_topics[i]] += chars

        words = len(docs[i].split(" "))
        count_doc_words[n_relevant_topics[i]] += words
    

    for i in range(len(count_relevant_topics)):
        if count_relevant_topics[i] > 0:
            count_doc_chars[i] = count_doc_chars[i] / count_relevant_topics[i]
            count_doc_words[i] = count_doc_words[i] / count_relevant_topics[i]

    print("Average number of characters for text passage with N relevant topics:", count_doc_chars)
    print("Average number of words for text passage with N relevant topics:", count_doc_words)


    output_path = f"{prob_path}/{df_name}_final_topics.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(final_diverse_topics_all, f)
    print(f"Successfully saved topics to {output_path}")

    # Stop the stopwatch
    end_time = time.time()
    
    # Calculate and format the elapsed time
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    print(f"Execution finished successfully!")
    print(f"Total processing time: {minutes} minutes and {seconds} seconds.")



#  Filter a ranked list of topics to ensure semantic diversity
def filter_topics_by_diversity(relevant_topics, jaccard_threshold, topics_dict):

    # relevant_topics: tuples (topic_id, topic_info)
    # Jaccard threshold: The maximum allowed word overlap (0.0 to 1.0)

    # Safety check: if the document has no topics, return an empty list
    if not relevant_topics:
        return []

    # Always keep the strongest topic (Index 0)
    diverse_topics = [relevant_topics[0]]

    # Evaluate every subsequent "challenger" topic
    for challenger_id, challenger_prob in relevant_topics[1:]:

        # Look up the words/weights for this specific topic ID
        challenger_info = topics_dict.get(challenger_id, [])
        
        # Safety check: skip if the topic_info is empty
        if not challenger_info:
            continue
            
        # Extract just the words (ignore the weights for the set operation)
        challenger_words = set([word for word, weight in challenger_info])
        is_redundant = False

        # Compare against all currently accepted "champion" topics
        for champion_id, champion_prob in diverse_topics:
            champion_info = topics_dict.get(champion_id, [])
            champion_words = set([word for word, weight in champion_info])

            # Calculate Jaccard Similarity: |Intersection| / |Union|
            intersection = len(challenger_words.intersection(champion_words))
            union = len(challenger_words.union(champion_words))
            jaccard_sim = intersection / union if union > 0 else 0

            # If it's too similar to ANY existing topic, we flag it as redundant
            if jaccard_sim > jaccard_threshold:
                is_redundant = True
                break  # No need to check other champions, it is already filtered

        # If it survived all checks, add it to the accepted list
        if not is_redundant:
            diverse_topics.append((challenger_id, challenger_prob))

    return diverse_topics


if __name__ == "__main__":

    seed = 42

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    main()