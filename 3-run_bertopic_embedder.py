import torch
import torch.nn.functional as F

from bertopic import BERTopic
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

from transformers.pipelines import pipeline
from transformers import BitsAndBytesConfig, AutoConfig, AutoModel, AutoTokenizer

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.backend import BaseEmbedder
import pickle

import gensim.corpora as corpora
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import utils

def main():


    # Define dataframe name and path
    df_name = 'arxiv'
    saved_path = f"saved/{df_name}_2"

    # Start time counter
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_datetime}] Execution started...")
    start_time = time.time()

    # Load documents
    df_text = pd.read_parquet("data/processed/parquet/"+ df_name + "_text.parquet")
    num_rows = len(df_text)
    print(f"Total documents: {num_rows}")

    # Extract the text column to a standard list
    docs = df_text['text'].to_list()

    # 3. Stack the individual embedding arrays into a single 2D NumPy matrix
    # Assuming your embeddings column is named 'embeddings'
    # df_embeddings = pd.read_parquet(df_name + "_embeddings.parquet")
    # final_embeddings_matrix = np.vstack(df_embeddings['embedding'].values)
    # print(f"Final matrix shape: {final_embeddings_matrix.shape}") 
    # Expected output: (number_of_texts, embedding_dimension)

    # Define embedding model
    embedding_model = F2LLMEmbedder()

    # Define training parameters
    umap_n_neighbors = 15
    umap_n_components = 5
    hdbscan_min_cluster_size = 30
    hdbscan_min_samples = 10
    min_df = 2
    min_topic_size = 10

    # Create submodels
    umap_model = UMAP(random_state=seed, n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=0.0, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, gen_min_span_tree=True, prediction_data=True,
    metric='euclidean', cluster_selection_method='eom')
    custom_pattern = r"\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b"
    vectorizer_model = CountVectorizer(token_pattern=custom_pattern, min_df=min_df, stop_words="english")
    # vectorizer_model = CountVectorizer(token_pattern="(?u)\\b(?!\\d+\\b)[\\w-]+\\b")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
    representation_model = KeyBERTInspired()

    # Build complete topic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=True,
        top_n_words=10,
        min_topic_size=min_topic_size
    )

    # Model fit
    # topics, probs = topic_model.fit_transform(docs, final_embeddings_matrix)
    topics, probs = topic_model.fit_transform(docs)

    # Get topic info
    # Topic -1 represents the outliers
    topic_info = topic_model.get_topic_info()
    n_topics = topic_info.shape[0]
    print("Number of topics:", n_topics-1)
    topic_info.to_csv(f"{saved_path}/{df_name}_topic_info.csv", index=False)

    # Get document info
    document_info = topic_model.get_document_info(docs)
    document_info.to_csv(f"{saved_path}/{df_name}_doc_info.csv", index=False)

    # Save fitted topic model
    topic_model.save(f"{saved_path}/{df_name}_model", 
    serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

    # Get and save topic dict
    topics_dict = topic_model.get_topics()
    with open(f"{saved_path}/{df_name}_topics_dict.pkl", "wb") as f:
        pickle.dump(topics_dict, f)

    # Build and save word list
    topics_word_list = utils.build_topic_word_list(topics_dict, 1)
    print(topics_word_list[0])
    utils.save_to_json(f"{saved_path}/{df_name}_topics_word_list.json", topics_word_list)

    # Save topic embeddings
    np.save(f"{saved_path}/{df_name}_embeddings.npy", topic_model.topic_embeddings_)
    # np.save(f"{saved_path}/{df_name}_topics.npy", topics)
    # np.save(f"{saved_path}/{df_name}_probs.npy", probs)


    # Compute approximate topic distributions for each document
    approx_distributions, _ = topic_model.approximate_distribution(docs)
    print(approx_distributions)
    print(np.sum(approx_distributions[0]))
    np.save(f"{saved_path}/{df_name}_approx_distributions.npy", approx_distributions)

    # Stop the stopwatch
    end_time = time.time()
    
    # Calculate and format the elapsed time
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    print(f"Execution finished successfully!")
    print(f"Total processing time: {minutes} minutes and {seconds} seconds.")

    # Compute coherence score
    u_mass = compute_coherence(docs, topic_model, metric='u_mass')
    print(f'u_mass: {u_mass:.4f}')

    npmi = compute_coherence(docs, topic_model, metric='c_npmi')
    print(f'npmi: {npmi:.4f}')

    c_v = compute_coherence(docs, topic_model, metric='c_v')
    print(f'c_v: {c_v:.4f}')

    # Compute DBCV score
    dbcv_score = topic_model.hdbscan_model.relative_validity_
    print(f"DBCV Score: {dbcv_score:.4f}")



def compute_coherence(docs, topic_model, metric='u_mass'):
    # https://github.com/MaartenGr/BERTopic/issues/90
    # For coherence computation
    cleaned_docs = topic_model._preprocess_text(docs)

    eval_vectorizer = topic_model.vectorizer_model
    analyzer = eval_vectorizer.build_analyzer()

    topics = topic_model.topics_

    words = eval_vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] for topic in range(len(set(topics))-1)]

    coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokens,
                    dictionary=dictionary,
                    coherence=metric,
                    topn=10
        )

    coherence_score = coherence_model.get_coherence()

    return coherence_score


class F2LLMEmbedder(BaseEmbedder):
    # Embedder class

    def __init__(
        self,
        instruction_prompt="Instruct: Identify the main topic of the text to facilitate semantic grouping.\nDocument: ",
        batch_size: int = 16,
    ):
        super().__init__()
        self.instruction_prompt = instruction_prompt
        self.batch_size = batch_size
        # self.model_id = "codefuse-ai/F2LLM-v2-1.7B"
        self.model_id = "codefuse-ai/F2LLM-v2-0.6B"
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",           # Normalized Float like r/LocalLLaMA
            bnb_4bit_use_double_quant=True       # Pls save even more VRAM 
        )   
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            quantization_config=self.quant_config,
            trust_remote_code=True,
            device_map="auto"
            )
    
    def get_embeddings(self, texts):
        with torch.no_grad():
            batch_size = len(texts)
            # the tokenizer will automatically add eos token
            tokenized_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            # --- PRINT TOKEN COUNT HERE ---
            # shape is [batch_size, sequence_length]
            num_tokens = tokenized_inputs.input_ids.shape[1]
            # print(f"Input batch has {num_tokens} tokens per sequence (including padding).")
            # ------------------------------
            last_hidden_state = self.model(**tokenized_inputs).last_hidden_state
            eos_positions = tokenized_inputs.attention_mask.sum(dim=1) - 1
            embeddings = last_hidden_state[torch.arange(batch_size, device=self.model.device), eos_positions]
            embeddings = F.normalize(embeddings, p=2, dim=1).to(torch.float32).cpu().numpy()
        return embeddings
    
    def embed(self, documents,verbose=False):
        embeddings_list = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            prompted_documents = [f"{self.instruction_prompt}{document}" for document in batch]
            emb = self.get_embeddings(prompted_documents)
            embeddings_list.append(emb)

        if len(embeddings_list) == 0:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = np.vstack(embeddings_list)
        return embeddings


if __name__ == "__main__":

    seed = 42

    # Define environment and random seed
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    main()