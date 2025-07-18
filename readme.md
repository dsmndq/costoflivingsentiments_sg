
# Singapore Cost of Living Sentiment Analysis

## 🎈 [**View the Live Demo Here**](https://costoflivingsentimentssg.streamlit.app/)

This project is an end-to-end data pipeline that scrapes Reddit comments related to the cost of living in Singapore, preprocesses the text data, and performs both sentiment analysis and topic modeling. The analysis uses two distinct models for sentiment—VADER (lexicon-based) and a pre-trained Transformer—and Latent Dirichlet Allocation (LDA) for topic modeling.

The project now includes an interactive web application built with Streamlit that allows users to perform sentiment analysis in real-time.

# Read about this project in [this blog post](https://medium.com/@desmond_57481/decoding-the-discourse-an-nlp-deep-dive-into-the-singapores-cost-of-living-conversation-a4a6010b426b)

## 📊 Features

-   **Reddit Data Scraping**: Dynamically scrapes comments from specified subreddits based on search queries and keywords.
-   **Secure Credential Management**: Uses a `config.ini` file to manage Reddit API credentials securely.
-   **Robust Text Preprocessing**: A comprehensive cleaning pipeline normalizes text by lowercasing, removing noise (URLs, mentions), filtering stopwords, and performing lemmatization.
-   **Interactive Web Application**: A user-friendly interface built with Streamlit to run sentiment analysis models directly in the browser.
-   **Multi-Faceted Analysis**:
    -   **VADER Sentiment Analysis**: A fast, lexicon-based model that provides detailed positive, neutral, and negative scores.
    -   **Transformer Sentiment Analysis**: A deep learning model (`finiteautomata/bertweet-base-sentiment-analysis`) for nuanced, context-aware sentiment classification.
    -   **Topic Modeling**: Identifies key themes within negative comments using Latent Dirichlet Allocation (LDA).
-   **Data & Visualization Output**: Each analysis script saves its results to a CSV file and generates visualizations.

## 📂 Project Structure

```
costoflivingsentiments_sg/
├── .gitignore
├── config.ini
├── scraping.py
├── preprocessing.py
├── detailed\_vader\_analysis.py
├── sentiment\_analysis\_transformer.py
├── topic\_modelling.py
├── app.py
├── requirements.txt
└── README.md
```

-   **`scraping.py`**: Connects to the Reddit API and scrapes comments.
-   **`preprocessing.py`**: Cleans the raw text data from the scraped comments.
-   **`detailed_vader_analysis.py`**: Performs sentiment analysis using VADER.
-   **`sentiment_analysis_transformer.py`**: Performs sentiment analysis using a Hugging Face Transformer model.
-   **`topic_modelling.py`**: Identifies topics in negative comments using LDA.
-   **`app.py`**: A self-contained script to launch an interactive Streamlit web application for real-time sentiment analysis.
-   **`config.ini`**: Configuration file for storing Reddit API credentials.
-   **`requirements.txt`**: A list of all Python libraries required to run the project.
-   **`.gitignore`**: Specifies which files to ignore in version control.

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dsmndq/costoflivingsentiments_sg.git](https://github.com/dsmndq/costoflivingsentiments_sg.git)
    cd costoflivingsentiments_sg
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the content below, then run the installation command. Note that `streamlit` has been added for the web app.
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
    scikit-learn
    streamlit
    ```
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Resources:**
    The scripts will automatically download the necessary NLTK packages on their first run.

5.  **Configure API Credentials (for local pipeline):**
    -   Create a `config.ini` file.
    -   Add your Reddit API `client_id` and `client_secret` to `config.ini`.

## ⚙️ Usage Workflow (Local Data Pipeline)

The original scripts are designed to be run in a specific order to form a complete data pipeline.

1.  **Scrape the Data:**
    ```bash
    python scraping.py
    ```
    *Output: `scraped_relevant_comments_praw.csv`*

2.  **Preprocess the Data:**
    ```bash
    python preprocessing.py
    ```
    *Output: `processed_corpus.csv`*

3.  **Run Sentiment Analysis (VADER or Transformer):**
    ```bash
    python detailed_vader_analysis.py
    # OR
    python sentiment_analysis_transformer.py
    ```

4.  **Run Topic Modeling:**
    ```bash
    python topic_modelling.py
    ```

## 🌐 Interactive App Deployment (Streamlit)

This project includes an interactive web application (`app.py`) that allows you to analyze text directly in your browser.

**The application is deployed and live here: [https://costoflivingsentimentssg.streamlit.app/](https://costoflivingsentimentssg.streamlit.app/)**

### Running the App Locally

1.  **Ensure Dependencies are Installed**: Make sure you have installed all packages from the updated `requirements.txt`, especially `streamlit`.
2.  **Run the App**: Execute the following command in your terminal from the project's root directory.
    ```bash
    streamlit run app.py
    ```
    Your browser should open with the application running locally.
