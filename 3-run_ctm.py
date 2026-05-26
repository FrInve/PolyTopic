import tomotopy as tp

import pandas as pd
import numpy as np
import pickle
import os
import utils
import json
import time
from datetime import datetime
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def main():

    # Define dataset name
    dataset_name = 'dbpedia'
    
    # Start time counter
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_datetime}] Execution started...")
    start_time = time.time()

    # Define path to documents and read dataframe
    doc_path = f"data/processed/parquet/{dataset_name}_text.parquet"
    df_doc = pd.read_parquet(doc_path)
    docs = df_doc['text'].tolist()

    # Define path to cleaned tokens and read dataframe
    tokens_path = f"data/processed/parquet/{dataset_name}_clean_tokens.parquet"
    df_tokens = pd.read_parquet(tokens_path)
    clean_tokens = df_tokens['clean_tokens'].tolist()
    print(f'Dataframe loaded with {len(df_doc)} documents')

    preprocessed_documents = []

    for arr in clean_tokens:
        sentence = " ".join(arr)
        preprocessed_documents.append(sentence)

    # Define number of topics to extract and number of training epochs
    n_topics = 500
    n_epochs = 50
    print('Number of topics to extract:', n_topics)

    # Create CTM model
    tp = TopicModelDataPreparation("all-mpnet-base-v2")
    training_dataset = tp.fit(text_for_contextual=docs, text_for_bow=preprocessed_documents)
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=n_topics, num_epochs=n_epochs)
    
    # Fit CTM model
    ctm.fit(training_dataset, verbose=True)

    topics_list = ctm.get_topic_lists(10)

    # Build dict with generated topics
    topics_dict = dict()

    for i in range(len(topics_list)):
        topics_dict[i] = []
        
        for topic in topics_list[i]:
            topics_dict[i].append((topic, 0))

    # Save dict
    utils.save_dict(f"saved/{dataset_name}_ctm/{dataset_name}_topics_dict.pkl", topics_dict)

    # Build and save word list
    topics_word_list = utils.build_topic_word_list(topics_dict, initial_index=0)
    utils.save_to_json(f"saved/{dataset_name}_ctm/{dataset_name}_topics_word_list.json", topics_word_list)

    # Extract and save document-topic distribution matrix
    full_distribution = ctm.get_doc_topic_distribution(training_dataset)
    print(full_distribution.shape)
    print(np.sum(full_distribution[0])) # Check if it is around 1

    np.save(f"saved/{dataset_name}_ctm/{dataset_name}_full_distribution.npy", full_distribution)


    # Stop the stopwatch
    end_time = time.time()
    
    # Calculate and format the elapsed time
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    print(f"Execution finished successfully!")
    print(f"Total processing time: {minutes} minutes and {seconds} seconds.")

    # Compute coherence score
    u_mass = compute_coherence(clean_tokens, ctm, metric='u_mass')
    print(f'u_mass: {u_mass:.4f}')

    npmi = compute_coherence(clean_tokens, ctm, metric='c_npmi')
    print(f'npmi: {npmi:.4f}')

    c_v = compute_coherence(clean_tokens, ctm, metric='c_v')
    print(f'c_v: {c_v:.4f}')




def compute_coherence(docs, ctm_model, metric='u_mass'):

    top_n = 10 # Number of words per topic to evaluate

    ctm_topics = ctm_model.get_topic_lists(top_n)

    gensim_dictionary = Dictionary(docs)

    coherence_model = CoherenceModel(
        topics=ctm_topics, 
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(seed)

    main()

