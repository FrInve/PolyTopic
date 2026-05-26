import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

def main():

    # Define dataset name and path
    dataset_name = 'amazoncat'
    data_path = f"data/processed/parquet/{dataset_name}_text.parquet"

    # Load parquet dataframe
    df = pd.read_parquet(data_path)
    print(f"Total rows in Parquet file: {len(df)}")

    # Download the stopwords dictionary (only need to run this once)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Create a new column 'clean_tokens' to hold the finalized lists
    df['clean_tokens'] = df['text'].apply(lambda x: clean_text(x, stop_words))
    df_clean = df[['clean_tokens']]

    # Save the tokens to a new dataset file
    df_clean.to_parquet(f'data/processed/parquet/{dataset_name}_clean_tokens.parquet', index=False)
    print('Tokenization complete and saved to Parquet.')


# Define the cleaning function
def clean_text(text, stop_words):
    # Safety check: if the row is missing/NaN, return an empty list
    if not isinstance(text, str):
        return []
    
    # Make text lowercase (to be done before stopword removal, because NLTK stopwords are all lowercase)
    text = text.lower()

    # Replace punctuation with space, but keep hyphens in composite words
    punc_no_hyphen = string.punctuation.replace('-', '')
    
    # Build a regex pattern with three rules:
    # [%s]      -> Match any standard punctuation (except hyphen)
    # (?<!\w)-  -> OR match a hyphen that does NOT have a word character before it
    # (?!\w)   -> OR match a hyphen that does NOT have a word character after it
    pattern = r'[%s]|(?<!\w)-|-(?!\w)' % re.escape(punc_no_hyphen)
    
    # 3. Replace matches with a space
    text = re.sub(pattern, ' ', text)
    
    # Replace punctuation with space
    # string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # This regex looks for any of those characters and swaps them for a space (" ")
    # text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    
    # Tokenize: .split() automatically handles multiple spaces and cuts the text into a list of words
    tokens = text.split()
    
    # Filter tokens (stopwords, numbers, length)
    cleaned_tokens = [
        word for word in tokens 
        if word not in stop_words      # 1. Must not be a stopword
        and not word.isdigit()         # 2. Must not be ONLY numbers (e.g., '1998')
        and len(word) >= 2             # 3. Must be 2 characters or longer
    ]
    
    return cleaned_tokens



if __name__ == "__main__":
    main()