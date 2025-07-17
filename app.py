import streamlit as st
import torch
from transformers import pipeline

# Import the preprocessing function from your project
from preprocessing import preprocess_social_media_text, download_nltk_resources

# --- 1. Model and Preprocessing Setup ---

# Use st.cache_resource to load the model only once and cache it for future runs.
# This significantly speeds up the app after the first load.
@st.cache_resource
def load_sentiment_pipeline():
    """
    Loads the Hugging Face sentiment analysis pipeline.
    The decorator ensures this function is run only once.
    """
    # Create a placeholder to show the user that the model is loading
    with st.spinner("Loading sentiment analysis model... This may take a moment."):
        # Use the same model as in your analysis script for consistency
        model_name = "finiteautomata/bertweet-base-sentiment-analysis"

        # Set device to GPU (0) if available, otherwise CPU (-1)
        device = 0 if torch.cuda.is_available() else -1

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device
        )
    return sentiment_pipeline

def standardize_label(label: str) -> str:
    """
    Standardizes the model's output label ('POS', 'NEG', 'NEU') to a
    user-friendly format ('Positive', 'Negative', 'Neutral').
    """
    if label == 'POS':
        return 'Positive'
    elif label == 'NEG':
        return 'Negative'
    elif label == 'NEU': # The model uses 'NEU' for Neutral
        return 'Neutral'
    return 'Unknown' # Fallback for any unexpected labels

@st.cache_resource
def setup_nltk():
    """Ensures NLTK resources are downloaded once per app session."""
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    download_nltk_resources()

# --- 2. Streamlit User Interface ---

def main():
    # Set up the Streamlit page configuration
    st.set_page_config(
        page_title="SG Cost of Living Sentiment",
        page_icon="🇸🇬",
        layout="centered"
    )

    st.title("🇸🇬 Singapore Cost of Living Sentiment Analysis")
    st.markdown("""
    Enter a comment about the cost of living in Singapore, and this app
    will predict the sentiment (Positive, Negative, or Neutral) using a
    fine-tuned `BERTweet` model.
    """)

    # Ensure NLTK resources are available (cached for efficiency)
    setup_nltk()

    # Load the cached model
    sentiment_pipeline = load_sentiment_pipeline()

    # --- User Input Area ---
    user_input = st.text_area(
        "Enter text for analysis:",
        "The rent is too high, but the food is amazing and affordable.",
        height=150
    )

    # --- Analysis Trigger ---
    if st.button("Analyze Sentiment"):
        if user_input and user_input.strip() != "":
            with st.spinner("Analyzing..."):
                # 1. Preprocess the text using your existing function
                cleaned_text = preprocess_social_media_text(user_input)

                # 2. Get prediction from the model (pipeline expects a list)
                result = sentiment_pipeline([cleaned_text])[0]
                sentiment = standardize_label(result['label'])
                score = result['score']

                # 3. Display the results in a user-friendly way
                st.subheader("Analysis Result")
                if sentiment == 'Positive':
                    st.success(f"**Sentiment: {sentiment}** (Confidence: {score:.2%})")
                elif sentiment == 'Negative':
                    st.error(f"**Sentiment: {sentiment}** (Confidence: {score:.2%})")
                else:
                    st.warning(f"**Sentiment: {sentiment}** (Confidence: {score:.2%})")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == '__main__':
    main()