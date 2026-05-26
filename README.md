# PolyTopic

Official repository for the paper **"PolyTopic: Extending Topic Modeling with Multi-Topic Assignment"**.

PolyTopic is a framework designed to handle documents with multiple topics by leveraging state-of-the-art topic modeling techniques and evaluating them using Large Language Models (LLMs).

## Pipeline Overview

The project follows a structured pipeline from data preparation to evaluation:

1.  **Data Cleaning** (`1-data_cleaning.py`): Preprocesses raw CSV datasets (Reuters, DBpedia, ArXiv), removing special characters and formatting text.
2.  **Tokenization** (`2-data_tokenization.py`): Performs text tokenization and stopword removal, essential for traditional topic models.
3.  **Topic Model Execution**:
    *   `3-run_bertopic_embedder.py`: Runs BERTopic using custom embeddings.
    *   `3-run_ctm.py`: Executes Contextualized Topic Models (CTM).
    *   `3-run_lda.py`: Runs Latent Dirichlet Allocation (LDA) models.
4.  **Topic Selection**:
    *   `4-select_topics_by_probs.py`: Filters topics based on probability thresholds.
    *   `5-select_topics_by_diversity.py`: Refines topic selection using diversity metrics.
5.  **Evaluation**:
    *   `6-evaluate_with_llm.py`: Uses LLMs (via OpenRouter API) to assess the relevance and coverage of assigned topics.
    *   `7-prepare_llm_results.py`: Post-processes and formats the LLM evaluation results.

## Getting Started

### Prerequisites

*   Python 3.10+
*   CUDA-compatible GPU (recommended for BERTopic and CTM)

### Installation

Clone the repository and install the dependencies:

```bash
cd Polytopic
pip install -r requirements.txt
```

### Usage

Each script is designed to be run sequentially. Ensure your data is placed in the `data/` directory as expected by the scripts.

Example of running the cleaning script:
```bash
python 1-data_cleaning.py
```

## Datasets

The framework has been tested on several standard datasets:
*   **ArXiv**: Research paper abstracts.
*   **DBpedia**: Wikipedia article summaries.
*   **Reuters**: News articles.

## Results and Analysis

Detailed analysis and visualizations can be found in the `notebooks/` directory:
*   `Arxiv.ipynb`, `DBpedia.ipynb`, `Reuters.ipynb`: Dataset-specific analysis.
*   `compute_diversity.ipynb`: In-depth look at topic diversity metrics.
*   `display_results.ipynb`: Visualization of final results.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
