import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import pickle
import utils
import matplotlib.pyplot as plt
from scipy.stats import halfnorm
import json




def main():

    # Start the mathematical stopwatch
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_datetime}] Execution started...")
    start_time = time.time()

    # Load data
    # Available dataframes: 'dbpedia', 'reuters', 'arxiv'
    df_name = 'dbpedia'
    df_text = pd.read_parquet("data/processed/parquet/"+ df_name + "_text.parquet")

    # with open(f"saved/{df_name}/{df_name}_topics_dict.pkl", "rb") as f:
    #     topics_dict = pickle.load(f)

    prob_path = f"saved/{df_name}_ctm"
    docs = df_text['text'].to_list()
    topic_distr = np.load(f"{prob_path}/{df_name}_full_distribution.npy")
    print(topic_distr.shape)
    
    # Determine automatically the potential relevant topics
    relevant_topics_for_all = select_topics_by_probability(docs, topic_distr)

    print(relevant_topics_for_all[0])

    # Count topics per document
    n_relevant_topics = []

    for i in range(len(relevant_topics_for_all)):
        n_relevant_topics.append(len(relevant_topics_for_all[i]))

    print("Max topics for document:", np.max(n_relevant_topics))
    print("Avg topics for document:", np.average(n_relevant_topics))


    count_relevant_topics = np.array([0]*11)

    for num in n_relevant_topics:
        count_relevant_topics[num] += 1

    print("Number of documents with n topics:", count_relevant_topics)

    count_doc_chars = np.array([0]*11)
    count_doc_words = np.array([0]*11)

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


    # Save retrieved relevant topics
    output_path = f"{prob_path}/{df_name}_relevant_topics.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(relevant_topics_for_all, f)
    print(f"Successfully saved topics to {output_path}")
    # utils.save_to_json(f"{prob_path}/{df_name}_relevant_topics", relevant_topics_for_all)

    # Stop the stopwatch
    end_time = time.time()
    
    # Calculate and format the elapsed time
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    print(f"Execution finished successfully!")
    print(f"Total processing time: {minutes} minutes and {seconds} seconds.")


# Select topics for all documents according to probability
def select_topics_by_probability(docs, topic_distr):

    relevant_topics_for_all = []

    for i in range(len(docs)):
        relevant_topics_for_doc = select_topics_per_doc(i, topic_distr[i])
        relevant_topics_for_all.append(relevant_topics_for_doc)
    
    return relevant_topics_for_all


# Select relevant topics per each document
def select_topics_per_doc(id_document, topic_distr_document):

    ranked_topics, ranked_probabilities = rank_topics(id_document, topic_distr_document, verbose=False)
    cumulative_remove_first = compute_cumulative_sum(ranked_probabilities)
    half_normal_pdf = compute_half_normal_pdf(cumulative_remove_first, enable_plot=False)
    n_chosen_topics = compute_n_relevant_topics(cumulative_remove_first, half_normal_pdf)
    relevant_topics = retrieve_relevant_topics_per_doc(ranked_topics, n_chosen_topics, ranked_probabilities)

    return relevant_topics
    

# Extracts and sorts the top N topics and their probabilities for a specific document
def rank_topics(id_document, topic_distr_document, top_n=10, verbose=False):

    # Get the raw array for the chosen document
    arr_topic_distr = np.array(topic_distr_document)

    # argsort() sorts ascending, [::-1] flips it to descending, [:top_n] grabs the top 10
    # This guarantees strict descending order
    ranked_topics = np.argsort(arr_topic_distr)[::-1][:top_n]

    # Vectorized mapping: instantly get all probabilities without a slow for-loop
    ranked_probabilities = arr_topic_distr[ranked_topics]


    if verbose:
        print(f"Document {id_document}:")
        for topic_id, prob in zip(ranked_topics, ranked_probabilities):
            print(f"Topic {topic_id}: Probability {prob}")

    return ranked_topics, ranked_probabilities


# Compute the cumulative sums of the probabilities
def compute_cumulative_sum(ranked_probabilities):
    arr_probs = np.array(ranked_probabilities)

    # Vectorized numpy:
    # 1. [::-1] flips the array backwards
    # 2. np.cumsum() calculates the normal running total
    # 3. [::-1] flips it back to your desired descending order
    cumulative_sum = np.cumsum(arr_probs[::-1])[::-1]
        
    return cumulative_sum

# Fits a half-normal reference distribution to the reverse recumulative sum, and returns discrete-scaled Gaussian values for the mathematical cutoff
def compute_half_normal_pdf(cumulative_sum_topics, enable_plot=False):

    y = np.array(cumulative_sum_topics)
    n_elements = len(y)
    
    if np.all(y == 0): 
        return np.zeros(n_elements)  # Return zeros if all cumulative sums are zero

    x_indices = np.arange(n_elements)

    # Calculate standard deviation based on your data
    mean_weighted = np.average(x_indices, weights=y)
    variance_weighted = np.average((x_indices - mean_weighted)**2, weights=y)
    std_weighted = np.sqrt(variance_weighted)

    if std_weighted == 0.0:
        # Create an array of pure zeros
        gaussian_scaled_discrete = np.zeros(n_elements)
        # Spike the very first index to match the height of y[0]
        gaussian_scaled_discrete[0] = y[0]
    
    else: 
        # Discrete calculation for the actual threshold cutoff
        g_raw_discrete = halfnorm.pdf(x_indices, loc=0, scale=std_weighted)
        gaussian_scaled_discrete = g_raw_discrete * (y[0] / np.max(g_raw_discrete))
        # print(gaussian_scaled_discrete) 

    # Smooth calculation exclusively for the plot
    if enable_plot:

        x_smooth = np.linspace(0, n_elements - 1, 50) # Increased to 100 for maximum smoothnes

        if std_weighted == 0.0:
            # Create a smooth curve that is just a flat line at y[0]
            gaussian_scaled_smooth = np.zeros(x_smooth.shape)
            # Spike the very first index to match the height of y[0]
            gaussian_scaled_smooth[0] = y[0]
    
        else: 
            g_raw_smooth = halfnorm.pdf(x_smooth, loc=0, scale=std_weighted)
            gaussian_scaled_smooth = g_raw_smooth * (y[0] / np.max(g_raw_discrete))

        # print(gaussian_scaled_smooth) 

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot 1: the original cumulative sum bars
        ax.bar(x_indices, y, alpha=0.6, label='Cumulative Sum', color='steelblue')
        
        # Plot 2: the smooth theoretical Gaussian curve (red line)
        ax.plot(x_smooth, gaussian_scaled_smooth, 'r-', linewidth=2, alpha=0.8, label='Half-Normal Curve (Theoretical)')
        
        # Plot 3: the discrete evaluation points (black dots)
        ax.plot(x_indices, gaussian_scaled_discrete, 'ko', markersize=6, zorder=3, label='Discrete Evaluation Points')

        # Formatting
        tick_labels = [str(n_elements - i) for i in x_indices]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Number of Included Topics (Reverse Order)', fontsize=12)
        ax.set_ylabel('Cumulative Probability Mass', fontsize=12)
        ax.set_title('Dynamic Thresholding: Signal vs. Noise', fontsize=14)
        
        # Set legend
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return gaussian_scaled_discrete


# Define the most relevant topics according to the comparison with the standard deviation
def compute_n_relevant_topics(cumulative_sum, half_normal_pdf):
    all_topics = len(cumulative_sum)
    n_topics = all_topics
    limit_topics = all_topics
    first_lower_index = 0

    for n in range(all_topics):
        if cumulative_sum[n] == 0.0:
            limit_topics = n
            # print("Limit topics:", limit_topics)
            break
    
    if limit_topics == 0:
        return 0


    for i in range(1, limit_topics):
        # print(f"Topic {i+1}: Cumulative Sum = {cumulative_sum[i]:.4f}, Half-Normal PDF = {half_normal_pdf[i]:.4f}")
        if cumulative_sum[i] < half_normal_pdf[i]:

            if first_lower_index == 0:
                n_topics = i
                first_lower_index = i
            
            if cumulative_sum[i] < 0.5 * half_normal_pdf[i]:
                return i
                
        else:
            if first_lower_index != 0:
                return i
    
    if first_lower_index == 0:
        return limit_topics

    return n_topics

# Build the list of relevant topics
def retrieve_relevant_topics_per_doc(ranked_topics, n_chosen_topics, ranked_probabilities):

    relevant_topics = []

    for i in range(n_chosen_topics):
        relevant_topics.append((ranked_topics[i], ranked_probabilities[i]))
        
    return relevant_topics




if __name__ == "__main__":

    seed = 42

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    main()