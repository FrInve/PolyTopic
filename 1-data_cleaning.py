#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re



def preprocess_text(file_name):
    # Define the file name
    # Reuters dataset: 'reuters',
    # DBpedia dataset: 'dbpedia',
    # ArXiv dataset: 'arxiv' 

    # Load the raw CSV file into a DataFrame, ensuring that the documents are contained in the column 'text'

    df = pd.read_csv(f'data/complete/{file_name}.csv')
    print(df.shape)

    # df = df.drop_duplicates(subset=['text'])

    # # Drop rows where 'text' is missing or empty
    # df = df.dropna(subset=['text']).reset_index(drop=True)
    # df = df[df['text'].astype(str).str.strip() != '']

    # Select only the 'text' column
    df_clean = df[['text']]

    # Convert to string
    df_clean['text'] = df_clean['text'].astype(str)

    # Remove newline characters
    df_clean['text'] = df_clean['text'].apply(lambda x:  re.sub(r'(\n)', ' ', x))

    # Remove invisible control characters (from \x00 to \x1F and \x7F), excluding \n(\x0A), \t(\x09) and \r(\x0D)
    regex_to_clean = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'
    df_clean['text'] = df_clean['text'].str.replace(regex_to_clean, '', regex=True)

    # Remove escaped double quotes
    df_clean['text'] = df_clean['text'].str.replace('\\"', '"', regex=False)

    # Remove extra spaces
    df_clean['text'] = df_clean['text'].apply(lambda text: " ".join(text.split()))

    # If Reuters dataset: remove the automatic signature 'Reuter' at the end of the text
    if 'reuters' in file_name:
        df_clean['text'] = df_clean['text'].str.replace(r'\s\bReuter\b\s*$', '', regex=True, flags=re.IGNORECASE)

    # Export the dataframe in parquet format
    df_clean.astype(
        {
            "text": "string",
        }
    ).to_parquet(f'data/processed/parquet/{file_name}_text.parquet', index=False)

    # Export the dataframe in CSV format
    df_clean.astype(
        {
            "text": "string",
        }
    ).to_csv(f'data/processed/csv/{file_name}_text.csv', index=False)

    print(df_clean.shape)
    print(df_clean.head())


if __name__ == "__main__":

    # Select from 'amazoncat', 'arxiv', 'dbpedia', 'reuters'
    file_name = 'amazoncat'
    preprocess_text(file_name)



