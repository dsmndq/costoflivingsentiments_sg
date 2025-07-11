# Singapore Cost of Living Sentiment Analysis

This project is an end-to-end data pipeline that scrapes Reddit comments related to the cost of living in Singapore, preprocesses the text data, and performs both sentiment analysis and topic modeling. The analysis uses two distinct models for sentiment—VADER (lexicon-based) and a pre-trained Transformer—and Latent Dirichlet Allocation (LDA) for topic modeling.

The goal is to analyze public sentiment on this key economic issue and identify the core topics of discussion, particularly within negative comments.

## 📊 Features

- **Reddit Data Scraping**: Dynamically scrapes comments from specified subreddits based on search queries and keywords.
- **Secure Credential Management**: Uses a `config.ini` file to manage Reddit API credentials securely, keeping them out of the source code.
- **Robust Text Preprocessing**: A comprehensive cleaning pipeline normalizes text by lowercasing, removing noise (URLs, mentions), filtering stopwords, and performing lemmatization.
- **Multi-Faceted Analysis**:
  - **VADER Sentiment Analysis**: A fast, lexicon-based model that provides detailed positive, neutral, and negative scores.
  - **Transformer Sentiment Analysis**: A deep learning model (`finiteautomata/bertweet-base-sentiment-analysis`) for nuanced, context-aware sentiment classification.
  - **Topic Modeling**: Identifies key themes and topics discussed within the negative comments using Latent Dirichlet Allocation (LDA).
- **Batch Processing**: The Transformer model analysis is performed in batches to handle large datasets efficiently and manage memory, especially on GPU.
- **Data & Visualization Output**: Each analysis script saves its results to a CSV file and generates visualizations (bar charts, pie charts) of the results.

## 📂 Project Structure

```
costoflivingsentiments_sg/
├── .gitignore
├── config.ini
├── scraping.py
├── preprocessing.py
├── detailed_vader_analysis.py
├── sentiment_analysis_transformer.py
├── topic_modelling.py
├── requirements.txt
└── README.md
```

- **`scraping.py`**: Connects to the Reddit API using credentials from `config.ini`, searches for relevant posts, and scrapes the comments.
- **`preprocessing.py`**: Cleans the raw text data from the scraped comments and prepares it for analysis.
- **`detailed_vader_analysis.py`**: Loads the processed corpus, performs sentiment analysis using VADER, and outputs results, top comments, and a bar chart.
- **`sentiment_analysis_transformer.py`**: Loads the processed corpus, performs sentiment analysis using a Hugging Face Transformer model, and outputs results and a bar chart.
- **`config.ini`**: Configuration file for storing Reddit API credentials. **(Not tracked by Git)**.
- **`requirements.txt`**: A list of all the Python libraries required to run the project.
- **`.gitignore`**: Specifies which files and directories to ignore in version control (e.g., credentials, data files, virtual environments).

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd costoflivingsentiments_sg
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the content below, then run the installation command.
    ```
    # requirements.txt
    pandas
    praw
    nltk
    torch
    transformers
    matplotlib
    tqdm
    configparser
    ```
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Resources:**
    The preprocessing and VADER scripts will automatically download the necessary NLTK packages (`punkt`, `stopwords`, `wordnet`, `vader_lexicon`) on their first run.

5.  **Configure API Credentials:**
    - Rename the `config.ini.example` to `config.ini` (or create it).
    - Open `config.ini` and replace the placeholder values with your actual Reddit API `client_id` and `client_secret`.

## ⚙️ Usage Workflow

The scripts are designed to be run in a specific order to form a complete data pipeline.

1.  **Scrape the Data:**
    Run the scraping script to collect comments from Reddit. You can adjust the subreddits, search query, and post limit inside the script.
    ```bash
    python scraping.py
    ```
    *Output: `scraped_relevant_comments_praw.csv`*

2.  **Preprocess the Data:**
    Clean the raw text to prepare it for analysis.
    ```bash
    python preprocessing.py
    ```
    *Output: `processed_corpus.csv`*

3.  **Run Sentiment Analysis:**
    You can run either or both of the analysis scripts. They both use `processed_corpus.csv` as input.

    **VADER Analysis:**
    ```bash
    python detailed_vader_analysis.py
    ```
    *Outputs: `vader_sentiment_results.csv`, `vader_sentiment_distribution.png`*

    **Transformer Analysis:**
    ```bash
    python sentiment_analysis_transformer.py
    ```
    *Outputs: `transformer_sentiment_results.csv`, `transformer_sentiment_distribution.png`*

 4.  **Run Topic Modeling:**
    See what topics are discussed in the negative comments.
    ```bash
    python topic_modelling.py
    ```
    *Outputs: `negative_topics_distribution.png`*


## 📈 Example Output

The analysis scripts will generate bar charts visualizing the distribution of sentiments, similar to this:

[VADER Sentiment Distribution](vader_sentiment_distribution.png)
[Transformer Sentiment Distribution](transformer_sentiment_distribution.png)
[Negative Topics Distribution](negative_topics_distribution.png)]

*(Note: You will need to run the scripts to generate your own images. These are placeholders.)*
