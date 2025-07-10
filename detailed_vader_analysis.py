import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from tqdm import tqdm
import textwrap

def download_nltk_vader():
    """
    Checks for and downloads the VADER lexicon if it's missing.
    """
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        print("[*] NLTK VADER lexicon is already available.")
    except LookupError:
        print("[*] Downloading NLTK VADER lexicon...")
        nltk.download('vader_lexicon')
        print("[*] Download complete.")

def analyze_vader_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> dict:
    """
    Analyzes the sentiment of a given text using a VADER analyzer.
    Returns a dictionary of all scores.
    """
    if not isinstance(text, str) or not text.strip():
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    return analyzer.polarity_scores(text)

def classify_sentiment(compound_score: float) -> str:
    """
    Classifies sentiment based on the VADER compound score.
    """
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def print_top_comments(df: pd.DataFrame):
    """
    Prints the top 3 most positive, negative, and neutral comments based on their scores.
    """
    print("\n" + "="*80)
    print("--- Top 3 Comments by Sentiment Score ---")
    print("="*80)

    # --- Top Positive Comments (by 'pos' score) ---
    top_positive = df.sort_values(by='vader_pos', ascending=False).head(3)
    print("\n--- Top 3 POSITIVE Comments (Highest 'pos' score) ---\n")
    for i, row in top_positive.reset_index().iterrows():
        print(f"  Comment {i+1} (Positive Score: {row['vader_pos']:.4f}):")
        wrapped_text = textwrap.fill(row['raw_text'], width=78, initial_indent='    ', subsequent_indent='    ')
        print(wrapped_text)
        print("-" * 20)

    # --- Top Negative Comments (by 'neg' score) ---
    top_negative = df.sort_values(by='vader_neg', ascending=False).head(3)
    print("\n--- Top 3 NEGATIVE Comments (Highest 'neg' score) ---\n")
    for i, row in top_negative.reset_index().iterrows():
        print(f"  Comment {i+1} (Negative Score: {row['vader_neg']:.4f}):")
        wrapped_text = textwrap.fill(row['raw_text'], width=78, initial_indent='    ', subsequent_indent='    ')
        print(wrapped_text)
        print("-" * 20)

    # --- Top Neutral Comments (by 'neu' score) ---
    top_neutral = df.sort_values(by='vader_neu', ascending=False).head(3)
    print("\n--- Top 3 NEUTRAL Comments (Highest 'neu' score) ---\n")
    for i, row in top_neutral.reset_index().iterrows():
        print(f"  Comment {i+1} (Neutral Score: {row['vader_neu']:.4f}):")
        wrapped_text = textwrap.fill(row['raw_text'], width=78, initial_indent='    ', subsequent_indent='    ')
        print(wrapped_text)
        print("-" * 20)

if __name__ == '__main__':
    # --- 1. Setup ---
    download_nltk_vader()
    vader_analyzer = SentimentIntensityAnalyzer()
    tqdm.pandas(desc="Analyzing VADER Sentiment")

    # --- Configuration ---
    INPUT_FILE = 'processed_corpus.csv'
    OUTPUT_FILE = 'vader_sentiment_results.csv'
    CHART_FILE = 'vader_sentiment_distribution.png'

    # --- 2. Load and Analyze Data ---
    print(f"\n--- Starting VADER sentiment analysis for {INPUT_FILE} ---")
    try:
        corpus_df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(corpus_df)} entries from the corpus.")
        corpus_df['cleaned_text'] = corpus_df['cleaned_text'].astype(str).fillna('')

        vader_results = corpus_df['cleaned_text'].progress_apply(lambda text: analyze_vader_sentiment(text, vader_analyzer))
        vader_df = pd.json_normalize(vader_results).add_prefix('vader_')
        corpus_df = pd.concat([corpus_df, vader_df], axis=1)
        corpus_df['vader_label'] = corpus_df['vader_compound'].apply(classify_sentiment)

        corpus_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved detailed VADER sentiment results to '{OUTPUT_FILE}'")

        # --- 3. Display Summary and Top Comments ---
        print("\n--- Overall Sentiment Distribution (VADER) ---")
        print(corpus_df['vader_label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
        print_top_comments(corpus_df)

        # --- 4. Visualize the Sentiment Distribution ---
        print("\n" + "="*80 + "\n--- Generating sentiment distribution plot ---")
        plt.style.use('ggplot')
        sentiment_counts = corpus_df['vader_label'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
        ax = sentiment_counts.plot(kind='bar', color=['#4CAF50', '#FFC107', '#F44336'], figsize=(10, 6), edgecolor='black')
        ax.bar_label(ax.containers[0], fmt='%d', label_type='edge', padding=3)
        plt.title('VADER Sentiment Distribution of Reddit Corpus', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment Category', fontsize=12)
        plt.ylabel('Number of Comments', fontsize=12)
        plt.xticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(CHART_FILE)
        print(f"Sentiment distribution plot saved to '{CHART_FILE}'")
        plt.show()

    except FileNotFoundError:
        print(f"\n[!] Error: Input file not found at '{INPUT_FILE}'. Please run 'preprocessing.py' first.")
    except Exception as e:
        print(f"\n[!] An unexpected error occurred: {e}")