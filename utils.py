import numpy as np
import pandas as pd
import os
import pickle
import json


def load_parquet_to_list(df_path):
    # Load parquet file to list
    df = pd.read_parquet(f'{df_path}.parquet')
    docs = df['text'].tolist()
    print(f'Number of documents loaded: {len(docs)}')

    return docs


def load_dict(dict_path):
    # Load pickle format dictionary
    with open(dict_path, "rb") as f:
        loaded_dict = pickle.load(f)
    
    print(loaded_dict[0])

    return loaded_dict

def save_dict(dict_path, data):
    # Save dictionary
    with open(dict_path, "wb") as f:
        pickle.dump(data, f)
    
    print('Dictionary saved')


def build_topic_word_list(topics_dict, initial_index):
    # Build word list from topic dict
    topic_word_dict = dict()

    for topic_id, words in topics_dict.items():
        topic_word_dict[topic_id] = ""
        for word, _ in words:
            topic_word_dict[topic_id] += word + " "
        
        topic_word_dict[topic_id] = topic_word_dict[topic_id].rstrip()

    topic_word_list = list(topic_word_dict.values())
    print(len(topic_word_list))
    
    return topic_word_list[initial_index:]


def save_to_json(file_path, data):
    # Save data to json
    with open(f"{file_path}", 'w') as f:
    # indent=2 is not needed but makes the file human-readable 
    # if the data is nested
        json.dump(data, f, indent=2) 
    
    print("File saved")


def load_json(file_path):
    # Load data from json
    with open(f"{file_path}", 'r') as f:
        data = json.load(f)
    
    return data






