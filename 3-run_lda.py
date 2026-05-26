import tomotopy as tp

import pandas as pd
import numpy as np
import pickle
import os
import utils
import json
import time
from datetime import datetime

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def main():

    # Define dataset name
    dataset_name = 'arxiv'
    
    # Start time counter
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_datetime}] Execution started...")
    start_time = time.time()

    # Define path to cleaned tokens and read dataframe
    tokens_path = f"data/processed/parquet/{dataset_name}_clean_tokens.parquet"
    df = pd.read_parquet(tokens_path)
    print(f'Dataframe loaded with {len(df)} documents')

    # Define number of topics
    n_topics = 200
    print('Number of topics to extract:', n_topics)

    # Create LDA model
    lda_model = tp.LDAModel(tw=tp.TermWeight.ONE, k=n_topics, seed=42)

    clean_text = df['clean_tokens'].tolist()

    for tokens in clean_text:

        tokens = list(tokens)
        
        # Keep safety check to ensure the document isn't empty
        if isinstance(tokens, list) and len(tokens) > 0:
            # Add document to LDA model
            lda_model.add_doc(words=tokens)
        
    print(f'Total documents added to the model: {len(lda_model.docs)}')

    # print(df['clean_tokens'][0])

    # Define training parameters
    print("Training...")
    lda_model.burn_in = 100
    n_epochs = 1000

    # Train model
    for i in range(0, n_epochs, 100):
        lda_model.train(100)
        print(f"Iteration {i+100}\tLog-likelihood: {lda_model.ll_per_word}")
    
    # Build dict with generated topics
    topics_dict = dict()

    for topic_id in range(lda_model.k):
        for word, prob in lda_model.get_topic_words(topic_id, top_n=10):
            if topic_id not in topics_dict:
                topics_dict[topic_id] = []
            topics_dict[topic_id].append((word, prob))
    
    # Save dict
    utils.save_dict(f"saved/{dataset_name}_lda/{dataset_name}_topics_dict.pkl", topics_dict)

    # Build and save word list
    topics_word_list = utils.build_topic_word_list(topics_dict, initial_index=0)
    utils.save_to_json(f"saved/{dataset_name}_lda/{dataset_name}_topics_word_list.json", topics_word_list)


    # Extract and save document-topic distribution matrix
    full_distribution = []

    for doc in list(lda_model.docs):
        topic_distribution = doc.get_topic_dist()
        full_distribution.append(topic_distribution)

    full_distribution = np.array(full_distribution)
    print(full_distribution.shape)
    print(np.sum(full_distribution[0])) # Check if it is around 1
    np.save(f"saved/{dataset_name}_lda/{dataset_name}_full_distribution.npy", full_distribution)


    # Stop the stopwatch
    end_time = time.time()
    
    # Calculate and format the elapsed time
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    print(f"Execution finished successfully!")
    print(f"Total processing time: {minutes} minutes and {seconds} seconds.")

    # Compute coherence score
    u_mass = compute_coherence(clean_text, lda_model, metric='u_mass')
    print(f'u_mass: {u_mass:.4f}')

    npmi = compute_coherence(clean_text, lda_model, metric='c_npmi')
    print(f'npmi: {npmi:.4f}')

    c_v = compute_coherence(clean_text, lda_model, metric='c_v')
    print(f'c_v: {c_v:.4f}')




def compute_coherence(docs, mdl, metric='u_mass'):

    top_n = 10 # Number of words per topic to evaluate
    tomotopy_topics = []

    for k in range(mdl.k):
        # get_topic_words returns a list of (word, probability) tuples
        topic_word_probs = mdl.get_topic_words(k, top_n=top_n)
        
        # Extract just the words
        topic_words = [word for word, prob in topic_word_probs]
        tomotopy_topics.append(topic_words)

    gensim_dictionary = Dictionary(docs)

    coherence_model = CoherenceModel(
        topics=tomotopy_topics, 
        texts=docs, 
        dictionary=gensim_dictionary, 
        coherence=metric,
        topn=10 
    )

    # Calculate the overall coherence score
    coherence_score = coherence_model.get_coherence()

    return coherence_score



if __name__ == "__main__":

    seed = 42

    # Define environment and random seed
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    main()
