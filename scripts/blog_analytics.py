import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Set styling for minimalist, professional visual presentation (shades of navy/plum/pink)
sns.set_theme(style="whitegrid")
PRIMARY_COLOR = "#4A2E80"  # Plum/Deep Purple
ACCENT_COLOR = "#E27B9B"   # Pink

def load_and_clean_data(filepath):
    """Loads scraped CSV and cleans up text content."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath} with {len(df)} records.")
        # Ensure text columns are filled and strings
        df['article_content'] = df['article_content'].fillna('').astype(str)
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run your scraper script first.")
        return None

def run_text_depth_analytics(df):
    """Calculates article lengths and proxy for emotional narrative depth."""
    print("\n--- Processing Text Length & Narrative Depth ---")
    df['word_count'] = df['article_content'].apply(lambda x: len(x.split()))
    avg_words = df['word_count'].mean()
    print(f"Average Article Narrative Length: {avg_words:.2f} words")
    return df

def run_readability_analytics(df):
    """Computes Flesch-Kincaid grade levels to evaluate reader accessibility."""
    print("\n--- Processing Readability Profiles ---")
    # Using textstat to evaluate the grade levels
    df['readability_score'] = df['article_content'].apply(
        lambda x: textstat.flesch_kincaid_grade(x) if len(x.strip()) > 10 else np.nan
    )
    print(f"Average Peer-to-Peer Readability Grade Level: {df['readability_score'].mean():.2f}")
    return df

def run_sentiment_analytics(df):
    """Performs VADER Sentiment Analysis tracking emotional articulation."""
    print("\n--- Running VADER Sentiment Analysis ---")
    sia = SentimentIntensityAnalyzer()
    
    # Calculate sentiment polarity scores
    df['sentiment_scores'] = df['article_content'].apply(lambda x: sia.polarity_scores(x))
    df['compound_sentiment'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])
    
    # Categorize base sentiments
    df['sentiment_category'] = pd.cut(
        df['compound_sentiment'], 
        bins=[-1.0, -0.05, 0.05, 1.0], 
        labels=['Negative/Pain Journey', 'Neutral/Informational', 'Positive/Empowering']
    )
    
    print("Sentiment Category Breakdown:")
    print(df['sentiment_category'].value_counts(normalize=True) * 100)
    return df

def generate_visual_analytics(df, output_img_name="digital_ethnography_analytics.png"):
    """Generates and saves professional dashboard figures of the analysis results."""
    print(f"\n--- Generating Analytics Dashboard Visuals -> {output_img_name} ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Sentiment Distribution (Reflecting emotional articulation)
    sns.histplot(df['compound_sentiment'], bins=20, kde=True, color=PRIMARY_COLOR, ax=axes[0])
    axes[0].set_title('Distribution of Patient Sentiment')
    axes[0].set_xlabel('Compound Sentiment Score (-1 Negative to +1 Positive)')
    axes[0].set_ylabel('Count of Posts')
    
    # Plot 2: Readability Levels (Accessibility for peers vs medical audience)
    sns.boxplot(y=df['readability_score'], color=ACCENT_COLOR, ax=axes[1])
    axes[1].set_title('Readability Profile (Flesch-Kincaid Grade)')
    axes[1].set_ylabel('Equivalent Grade Level')
    
    # Plot 3: Categorized Sentiments Breakdown
    df['sentiment_category'].value_counts().plot(kind='bar', color=[PRIMARY_COLOR, ACCENT_COLOR, '#8B78A5'], ax=axes[2])
    axes[2].set_title('Lived Experience Narrative Categorization')
    axes[2].set_ylabel('Number of Blog Articles')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_img_name, dpi=300)
    print("Visual analytics dashboard saved successfully.")

def main():
    # Replace filename with your actual extracted blog file (e.g., 'itsjustabadday_full_data.csv')
    csv_file = "itsjustabadday_full_data.csv" 
    
    df = load_and_clean_data(csv_file)
    if df is not None:
        df = run_text_depth_analytics(df)
        df = run_readability_analytics(df)
        df = run_sentiment_analytics(df)
        
        # Save enriched analytical dataset
        enriched_filename = "analyzed_blog_patient_insights.csv"
        df.to_csv(enriched_filename, index=False)
        print(f"\nSaved comprehensively analyzed data matrix to '{enriched_filename}'")
        
        # Generate final metrics dashboard plots
        generate_visual_analytics(df)

if __name__ == "__main__":
    main()
