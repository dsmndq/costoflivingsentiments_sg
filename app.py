import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Singapore Cost of Living Sentiment Analysis",
    page_icon="🇸🇬",
    layout="wide"
)


# --- NLTK Resource Management ---

# Use a flag to ensure NLTK resources are downloaded only once.
@st.cache_resource
def ensure_nltk_resources():
    """Downloads necessary NLTK resources if they are not already present."""
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'sentiment/vader_lexicon': 'vader_lexicon'
    }
    for path, package_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package_id)

# Call the function to ensure resources are available
ensure_nltk_resources()


# --- Text Preprocessing Function (from your preprocessing.py) ---

# Pre-initialize lemmatizer and stop words for efficiency
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))
# Add custom stopwords relevant to the project's context
CUSTOM_STOPWORDS = {'singapore', 'sg', 'like', 'get', 'one', 'also', 'would', 'really', 'im', 'u'}
STOP_WORDS.update(CUSTOM_STOPWORDS)

def preprocess_social_media_text(text: str) -> str:
    """
    Cleans and preprocesses raw text from social media for NLP tasks.
    The pipeline includes lowercasing, noise removal, tokenization,
    stopword removal, and lemmatization.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower() # Lowercasing
    text = re.sub(r'https://\S+|http\S+|www\S+', '', text) # Remove URLs
    text = re.sub(r'u/\w+|r/\w+', '', text) # Remove Reddit mentions
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    tokens = text.split()
    # Remove stopwords and lemmatize
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    lemmatized_tokens = [LEMMATIZER.lemmatize(word) for word in filtered_tokens]

    return ' '.join(lemmatized_tokens)


# --- Model Loading with Caching ---

@st.cache_resource
def load_vader_model():
    """Loads the VADER sentiment intensity analyzer."""
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_transformer_model():
    """
    Loads the pre-trained Transformer model for sentiment analysis.
    This model is specified in the project's readme.
    """
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    # The device=-1 argument ensures the model runs on CPU, which is suitable for Streamlit Cloud's free tier.
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=-1)

# --- Main Application UI ---

st.title("🗣️ Singapore Cost of Living Sentiment Analysis")
st.markdown("""
This app analyzes the sentiment of text about the cost of living in Singapore.
Enter some text below and choose a model to see the sentiment result.
""")

# --- User Input and Model Selection ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Your Text")
    user_input = st.text_area("Enter the text you want to analyze:", "The cost of living is getting very high, but the food is still amazing.", height=150)

    model_choice = st.selectbox(
        "🧠 Choose the Analysis Model:",
        ("Transformer (bertweet-base)", "VADER (Lexicon-based)")
    )

analyze_button = st.button("Analyze Sentiment", type="primary")


# --- Analysis and Results Display ---
with col2:
    st.subheader("📊 Analysis Results")

    if analyze_button:
        if user_input:
            with st.spinner("Analyzing..."):
                # Step 1: Preprocess the text
                cleaned_text = preprocess_social_media_text(user_input)
                st.write("**Preprocessed Text:**")
                st.info(f"'{cleaned_text}'")

                # Step 2: Run the selected model
                if model_choice == "VADER (Lexicon-based)":
                    vader_model = load_vader_model()
                    vader_scores = vader_model.polarity_scores(cleaned_text)

                    # Determine final sentiment
                    if vader_scores['compound'] >= 0.05:
                        final_sentiment = "Positive"
                        st.success("Sentiment: Positive ✅")
                    elif vader_scores['compound'] <= -0.05:
                        final_sentiment = "Negative"
                        st.error("Sentiment: Negative ❌")
                    else:
                        final_sentiment = "Neutral"
                        st.warning("Sentiment: Neutral ➖")

                    # Display VADER scores
                    st.write("**VADER Score Breakdown:**")
                    scores_df = pd.DataFrame([vader_scores]).T
                    scores_df.columns = ['Score']
                    st.dataframe(scores_df, use_container_width=True)


                elif model_choice == "Transformer (bertweet-base)":
                    transformer_model = load_transformer_model()
                    transformer_result = transformer_model(cleaned_text)
                    label = transformer_result[0]['label']
                    score = transformer_result[0]['score']

                    # Display Transformer result
                    if label == 'POS':
                        st.success(f"Sentiment: Positive ✅ (Score: {score:.2f})")
                    elif label == 'NEG':
                        st.error(f"Sentiment: Negative ❌ (Score: {score:.2f})")
                    else:
                        st.warning(f"Sentiment: Neutral ➖ (Score: {score:.2f})")

                    st.write("**Model Output:**")
                    st.json(transformer_result)

        else:
            st.warning("Please enter some text to analyze.")

st.markdown("---")
st.write("Project by Desmond Quek. Read more about it on [Medium](https://medium.com/@desmond_57481/decoding-the-discourse-an-nlp-deep-dive-into-the-singapores-cost-of-living-conversation-a4a6010b426b).")