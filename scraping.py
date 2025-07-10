
import praw
import pandas as pd
import time
import re

def initialize_reddit_client():
    """
    Initializes and returns a PRAW Reddit instance using your credentials.
    
    Returns:
        praw.Reddit: An authenticated Reddit instance.
    """

    # IMPORTANT: Hardcoding credentials is not recommended for production.
    # Consider using environment variables or a config file.
    CLIENT_ID = "YOUR_CLIENT_ID"
    CLIENT_SECRET = "YOUR_CLIENT_SECRET"
    USER_AGENT = "YOUR_USER_AGENT"
    
    # --- End of Credentials Section ---

    if CLIENT_ID == "YOUR_CLIENT_ID" or CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        print("[!] ERROR: Please update the script with your Reddit API credentials.")
        return None

    print("[*] Initializing Reddit API client...")
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )
    print("[*] Reddit client initialized successfully (read-only mode).")
    return reddit

def scrape_with_praw(reddit: praw.Reddit, subreddits: list[str], query: str, limit_per_sub: int = 5) -> list[str]:
    """
    Uses PRAW to search for a query across multiple subreddits and scrapes comments.

    Args:
        reddit (praw.Reddit): The authenticated PRAW instance.
        subreddits (list[str]): A list of subreddit names to search in.
        query (str): The search query.
        limit_per_sub (int): The max number of posts to process per subreddit.

    Returns:
        list[str]: A list of all scraped comments.
    """
    all_comments = []
    print("\n--- Phase 1: Finding and Scraping Relevant Threads ---")
    for sub_name in subreddits:
        print(f"[*] Searching in r/{sub_name} for query: '{query}'")
        subreddit = reddit.subreddit(sub_name)
        
        # Use the search method to find relevant submissions
        try:
            search_results = subreddit.search(query, sort='relevance', time_filter='all', limit=limit_per_sub)
            
            for submission in search_results:
                print(f"    -> Processing thread: '{submission.title[:50]}...'")
                # This line is crucial. It expands all "MoreComments" objects.
                submission.comments.replace_more(limit=None)
                
                # Iterate through all comments in the thread
                for comment in submission.comments.list():
                    # Ensure the comment has a body and isn't deleted
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        all_comments.append(comment.body)
                
                time.sleep(0.5) # Be a good citizen
        except Exception as e:
            print(f"    [!] Could not search in r/{sub_name}. It might be private or banned. Error: {e}")
            
    print(f"\n[*] Scraped a total of {len(all_comments)} comments before filtering.")
    return all_comments

def filter_comments_by_keywords(comments: list[str], keywords: list[str]) -> list[str]:
    """
    Filters a list of comments, keeping only those that contain at least one of the specified keywords.
    """
    print("\n--- Phase 2: Filtering Comments Based on Keywords ---")
    relevant_comments = []
    pattern = re.compile('|'.join(keywords), re.IGNORECASE)
    
    for comment in comments:
        if pattern.search(comment):
            relevant_comments.append(comment)
            
    print(f"[*] Kept {len(relevant_comments)} out of {len(comments)} comments after filtering.")
    return relevant_comments

def save_comments_to_csv(comments: list[str], filename: str):
    """Saves a list of comments to a single-column CSV file."""
    if not comments:
        print("[!] No comments to save.")
        return
    df = pd.DataFrame(comments, columns=['raw_text'])
    df.drop_duplicates(inplace=True)
    df.to_csv(filename, index=False)
    print(f"[*] Saved {len(df)} unique comments to '{filename}'")

if __name__ == '__main__':
    # --- Configuration ---
    SEARCH_QUERY = "cost of living"
    SINGAPORE_SUBREDDITS = ["singapore", "askSingapore", "SingaporeRaw", "singaporefi"]
    RELEVANT_KEYWORDS = [
        "salary", "expensive", "price", "cost", "gst", "inflation",
        "budget", "spend", "bills", "groceries", "rent", "bto", "hdb", "cpf"
    ]
    OUTPUT_CSV_FILE = "scraped_relevant_comments_praw.csv"

    # --- Scraping and Filtering Pipeline ---
    # Step 1: Initialize the PRAW client
    reddit_client = initialize_reddit_client()

    if reddit_client:
        # Step 2: Scrape comments using PRAW
        scraped_comments = scrape_with_praw(reddit_client, SINGAPORE_SUBREDDITS, SEARCH_QUERY, limit_per_sub=5)

        # Step 3: Filter the combined list of comments for relevance
        filtered_comments = filter_comments_by_keywords(scraped_comments, RELEVANT_KEYWORDS)

        # Step 4: Save the final, relevant comments
        save_comments_to_csv(filtered_comments, OUTPUT_CSV_FILE)
