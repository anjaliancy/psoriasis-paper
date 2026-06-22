import os
import re
import ast
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import praw
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# --- VISUAL THEME CONFIGURATION ---
PRIMARY_COLOR = "#3B2A54"   # Rich Deep Plum
ACCENT_COLOR = "#E27B9B"    # Dusty Pink
NAVY_NEUTRAL = "#1A365D"    # Slate Navy
PALETTE = [PRIMARY_COLOR, ACCENT_COLOR, NAVY_NEUTRAL, "#8B78A5"]

sns.set_theme(style="whitegrid", palette=sns.color_palette(PALETTE))
plt.rcParams.update({
    'font.size': 11,
    'axes.titlepad': 12,
    'figure.titlesize': 14,
    'axes.edgecolor': '#cccccc',
    'savefig.dpi': 300
})
warnings.filterwarnings('ignore')


def init_environment_and_nltk():
    """Download environment requirements seamlessly."""
    for res in ['punkt', 'stopwords', 'punkt_tab']:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"NLTK download skipped/warning for {res}: {e}")


# --- SECTION 1: DATA EXTRACTION MATRIX (SECURED) ---

def connect_reddit_api():
    """
    Initializes the PRAW API Client instance safely using environment variables.
    To set these up locally or in your environment, run:
    export REDDIT_CLIENT_ID="your_id"
    export REDDIT_CLIENT_SECRET="your_secret"
    export REDDIT_USER_AGENT="your_agent"
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID_HERE")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")
    user_agent = os.getenv("REDDIT_USER_AGENT", "DigitalEthnographyBot/2.0")
    
    if "HERE" in client_id or "HERE" in client_secret:
        print("Warning: Please set your REDDIT environment variables or replace placeholders.")
        
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

def harvest_subreddit_cohorts(reddit_client, subreddits, target_limit=1000, search_term=None):
    """
    Unified extraction matrix maximizing throughput across specified cohorts.
    Incorporates both raw hot/top feeds and strict targeted topical searches.
    """
    posts_accumulator = []
    comments_accumulator = []
    
    init_environment_and_nltk()
    print(f"Starting Data Harvest Target: {target_limit} posts per community.")

    for sub_name in subreddits:
        print(f"-> Processing Ecosystem: r/{sub_name}")
        try:
            subreddit = reddit_client.subreddit(sub_name)
            
            # Switch modes between generic deep scrape vs localized diagnostic search string
            if search_term:
                stream = subreddit.search(query=search_term, limit=target_limit)
            else:
                stream = subreddit.top(limit=target_limit)

            for submission in stream:
                # Conditional text evaluation if keyword query target mapping is specified
                if search_term:
                    title_text = submission.title.lower()
                    body_text = submission.selftext.lower() if submission.selftext else ""
                    if search_term.lower() not in title_text and search_term.lower() not in body_text:
                        continue
                
                # Metadata Structuring
                posts_accumulator.append({
                    "post_id": submission.id,
                    "title": str(submission.title),
                    "author": submission.author.name if submission.author else "[deleted]",
                    "upvotes": int(submission.score),
                    "num_comments": int(submission.num_comments),
                    "url": str(submission.url),
                    "created_utc": pd.to_datetime(submission.created_utc, unit='s'),
                    "subreddit": sub_name
                })

                # Inline Comment Extraction Flow (Rate-Limit Defensive Expansion)
                try:
                    submission.comments.replace_more(limit=0)
                    for top_comment in submission.comments.list():
                        if hasattr(top_comment, 'body') and top_comment.body:
                            comments_accumulator.append({
                                "post_id": submission.id,
                                "subreddit": sub_name,
                                "comment_id": top_comment.id,
                                "comment_author": top_comment.author.name if top_comment.author else "[deleted]",
                                "comment_body": str(top_comment.body),
                                "comment_upvotes": int(top_comment.score)
                            })
                except Exception as comment_err:
                    continue  # Fail-safe protection for transient connection drops

        except Exception as e:
            print(f"Execution warning on r/{sub_name}: {e}")
            
    return pd.DataFrame(posts_accumulator), pd.DataFrame(comments_accumulator)


# --- SECTION 2: READABILITY METRICS MANIFEST PIPELINE ---

def run_manuscript_readability(df, target_text_col='title'):
    """Calculates full linguistic density distribution indexes requested by the methodology."""
    print(f"\n--- Processing Readability Profiles over [{target_text_col}] ---")
    if df.empty:
        return df
        
    df[target_text_col] = df[target_text_col].astype(str).fillna('')
    
    # Calculate explicit structural indexes
    df['flesch_kincaid_grade'] = df[target_text_col].apply(textstat.flesch_kincaid_grade)
    df['flesch_reading_ease'] = df[target_text_col].apply(textstat.flesch_reading_ease)
    df['dale_chall_score'] = df[target_text_col].apply(textstat.dale_chall_readability_score)
    df['gunning_fog_index'] = df[target_text_col].apply(textstat.gunning_fog)
    
    # Display Baseline Ecosystem Aggregations
    summary = df.groupby('subreddit')['flesch_kincaid_grade'].mean().reset_index()
    print("Mean Reading Level Profiling per Target Cohort:")
    print(summary.to_string(index=False))
    
    return df


# --- SECTION 3: HYBRID SENTIMENT CASCADING ANALYSIS ---

class HybridSentimentEngine:
    """Implements cascading sentiment logic to balance computational efficiency and accuracy."""
    def __init__(self):
        print("Warming up VADER Lexicon and RoBERTa Transformers...")
        self.vader = SentimentIntensityAnalyzer()
        # Initializing local Transformer accuracy layer mapping
        self.roberta = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1  # Swap to 0 if executing on verified PyTorch CUDA environments
        )
        self.label_map = {"neutral": "Neutral", "positive": "Positive", "negative": "Negative"}

    def compute_row_sentiment(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "Neutral", 0.0, "N/A"

        # Pass 1: VADER Evaluation
        vader_res = self.vader.polarity_scores(text)
        compound = vader_res['compound']

        # Confident Assertion Strategy
        if abs(compound) >= 0.6:
            label = "Positive" if compound > 0 else "Negative"
            return label, float(abs(compound)), "VADER"

        # Pass 2: Contextual Transformer Fallback for Nuanced Lived-Experience Narratives
        try:
            # Enforce max block limitations safely
            truncated_text = " ".join(text.split()[:350])
            rob_res = self.roberta(truncated_text, truncation=True, max_length=512)[0]
            mapped_label = self.label_map.get(rob_res['label'].lower(), "Neutral")
            return mapped_label, float(round(rob_res['score'], 4)), "RoBERTa"
        except Exception:
            # Safe structural fallback
            label = "Neutral" if abs(compound) < 0.05 else ("Positive" if compound > 0 else "Negative")
            return label, float(abs(compound)), "VADER_Fallback"


def execute_sentiment_pipeline(comments_df):
    """Maps execution loops across observations cleanly via array optimizations."""
    print("\n--- Initiating Lived Experience Emotional Articulation Mapping ---")
    if comments_df.empty:
        return comments_df
        
    engine = HybridSentimentEngine()
    
    # Transform computations systematically
    results = [engine.compute_row_sentiment(text) for text in comments_df['comment_body']]
    
    # Unpack values securely back into dataframe layout
    comments_df['sentiment_label'] = [r[0] for r in results]
    comments_df['sentiment_score'] = [r[1] for r in results]
    comments_df['sentiment_method'] = [r[2] for r in results]
    
    print("\nGlobal Sentiment Target Categorization Breakdown:")
    print(comments_df['sentiment_label'].value_counts(normalize=True) * 100)
    return comments_df


# --- SECTION 4: INTEGRATED EXPORT & ANALYSIS SUMMARY GENERATION ---

def generate_analytical_dashboard(posts_df, comments_df):
    """Saves metrics, prints summaries, and outputs high-resolution analytical plots."""
    print("\n--- Outputting Visualization Plots and Structural Matrix Archives ---")
    
    # Export datasets
    posts_df.to_csv('psoriasis-all-reddit-posts.csv', index=False)
    comments_df.to_csv('psoriasis-all-reddit-comments.csv', index=False)
    print("Files preserved: [psoriasis-all-reddit-posts.csv], [psoriasis-all-reddit-comments.csv]")
    
    # Create Panel Plot Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Visual 1: Metric Distributions per Community
    sns.boxplot(x='subreddit', y='upvotes', data=posts_df, ax=axes[0], color=PRIMARY_COLOR, width=0.4)
    axes[0].set_yscale('log')
    axes[0].set_title('Cross-Cohort Post Virality Profile', fontweight='semibold')
    axes[0].set_xlabel('Patient Ecosystem Subreddit')
    axes[0].set_ylabel('Upvote Magnitude (Log Scale)')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Visual 2: Linguistic Accessibility Ranges
    sns.boxplot(x='subreddit', y='flesch_kincaid_grade', data=posts_df, ax=axes[1], color=ACCENT_COLOR, width=0.4)
    axes[1].set_title('Readability Profile of Psoriasis-Related Content', fontweight='semibold')
    axes[1].set_xlabel('Patient Ecosystem Subreddit')
    axes[1].set_ylabel('Flesch-Kincaid Equivalent Grade Level')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('reddit_ethnography_dashboard.png', dpi=300)
    plt.show()


# --- CONTROL SEQUENCE DRIVER ---

def main():
    target_subreddits = [
        'Psoriasis',
        'PsoriaticArthritis',
        'ScalpPsoriasis',
        'PsoriasisRemedies',
        'DermatologyQuestions'
    ]
    
    # Setup Connection using secure environment variable architecture
    reddit = connect_reddit_api()
    
    # Execute extraction on target communities
    posts_df, comments_df = harvest_subreddit_cohorts(
        reddit_client=reddit, 
        subreddits=target_subreddits, 
        target_limit=500,  
        search_term="Psoriasis"
    )
    
    if not posts_df.empty:
        # Enrich Data Model with Manuscript Analytics
        posts_df = run_manuscript_readability(posts_df, target_text_col='title')
        comments_df = execute_sentiment_pipeline(comments_df)
        
        # Output Visual Ensembles
        generate_analytical_dashboard(posts_df, comments_df)
        print("\n[Execution Pipeline Completed Terminating Process Successfully]")
    else:
        print("Extraction completed empty. Verify active internet or credentials strings.")

if __name__ == "__main__":
    main()
