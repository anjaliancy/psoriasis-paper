import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import textstat  # To implement the manuscript's Readability Profile

# --- CONFIGURATION & STYLING ---
# Professional, minimalist visual identity using deep plum, dusty pink, and clean navy accents
COLOR_PALETTE = ["#3B2A54", "#E27B9B", "#1A365D", "#8B78A5"]
sns.set_theme(style="whitegrid", palette=sns.color_palette(COLOR_PALETTE))
plt.rcParams.update({
    'font.size': 11,
    'axes.titlepad': 14,
    'figure.titlesize': 14,
    'axes.edgecolor': '#cccccc'
})

def download_nltk_resources():
    """Download required NLTK components securely."""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab', 'vader_lexicon']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Warning downloading {res}: {e}")
    print("NLTK packages verified successfully.")


# --- SECTION 1: DATA PREPROCESSING & CLEANING ---

def light_clean_text(text):
    """Performs raw string structural cleaning (URLs, HTML, Whitespace)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'&\w+;', '', text)
    return " ".join(text.split())

def advanced_nlp_preprocess(series):
    """Applies systematic lowercase normalization, punctuation removal, and stopword filtering."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_list = []
    for text in series:
        # Base normalization
        clean_str = str(text).lower().translate(str.maketrans('', '', string.punctuation))
        clean_str = ''.join([c for c in clean_str if not c.isdigit()])
        
        # Tokenization & Lemmatization
        tokens = word_tokenize(clean_str)
        filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        processed_list.append(filtered)
        
    return processed_list


# --- SECTION 2: QUANTITATIVE META & ENGAGEMENT METRICS ---

def analyze_and_plot_engagement(meta1, meta2):
    """Combines metrics datasets and plots platform distribution metrics."""
    print("\n--- Processing Cross-Platform Engagement Profiles ---")
    df1 = meta1[['statistics.like_count', 'statistics.comment_count']].copy()
    df1['platform'] = 'Facebook'
    
    df2 = meta2[['statistics.like_count', 'statistics.comment_count']].copy()
    df2['platform'] = 'Instagram'
    
    combined = pd.concat([df1, df2], ignore_index=True)
    
    metrics = ['statistics.like_count', 'statistics.comment_count']
    titles = ['Distribution of Like Counts per Platform', 'Distribution of Comment Counts per Platform']
    ylabels = ['Like Count (Log Scale)', 'Comment Count (Log Scale)']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, metric in enumerate(metrics):
        sns.boxplot(x='platform', y=metric, data=combined, ax=axes[i], width=0.5)
        axes[i].set_yscale('log')
        axes[i].set_title(titles[i], fontweight='semibold')
        axes[i].set_xlabel('Platform')
        axes[i].set_ylabel(ylabels[i])
        axes[i].grid(True, which="both", ls="--", c='0.85')
        
    plt.tight_layout()
    plt.show()
    return combined

def analyze_temporal_trends(meta1, meta2):
    """Maps structural and timeline changes across collection months."""
    print("\n--- Processing Temporal Content Volume Trends ---")
    for df, platform in zip([meta1, meta2], ['Facebook', 'Instagram']):
        df['creation_time'] = pd.to_datetime(df['creation_time'])
        df['year_month'] = df['creation_time'].dt.to_period('M')
        
    p_m1 = meta1.groupby('year_month').size().reset_index(name='post_count')
    p_m1['platform'] = 'Facebook'
    
    p_m2 = meta2.groupby('year_month').size().reset_index(name='post_count')
    p_m2['platform'] = 'Instagram'
    
    combined_timeline = pd.concat([p_m1, p_m2])
    combined_timeline['year_month'] = combined_timeline['year_month'].dt.to_timestamp()
    
    plt.figure(figsize=(11, 5.5))
    sns.lineplot(x='year_month', y='post_count', hue='platform', data=combined_timeline, marker='o', linewidth=2.5)
    plt.title('Number of Patient Posts per Month (Facebook vs. Instagram)', fontweight='semibold')
    plt.xlabel('Timeline Record (Year-Month)')
    plt.ylabel('Aggregated Post Count')
    plt.grid(True, which="both", ls="--", c='0.85')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Isolate absolute peak periods
    for platform in ['Facebook', 'Instagram']:
        sub = combined_timeline[combined_timeline['platform'] == platform]
        if not sub.empty:
            peak_row = sub.loc[sub['post_count'].idxmax()]
            print(f"Peak Activity Month [{platform}]: {peak_row['year_month'].strftime('%Y-%m')} ({peak_row['post_count']} posts)")


# --- SECTION 3: TEXT MINING & TOPRIC MODELING (LDA / BERTopic) ---

def execute_lda_modeling(tokens_list, num_topics=5):
    """Trains a Latent Dirichlet Allocation structure and visualizes thematic profiles."""
    print(f"\n--- Constructing LDA Generative Top-Word Profiles ({num_topics} Topics) ---")
    dictionary = Dictionary(tokens_list)
    corpus = [dictionary.doc2bow(text) for text in tokens_list]
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    
    # Horizontal validation distribution plotting
    for i, topic in lda_model.show_topics(formatted=False, num_topics=num_topics, num_words=10):
        fig, ax = plt.subplots(figsize=(7, 4))
        words = [word for word, _ in topic]
        weights = [weight for _, weight in topic]
        
        ax.barh(words, weights, color='#8B78A5', edgecolor='#4A3B63', height=0.6)
        ax.set_title(f'Topic {i+1} Principal Keyword Metrics', fontweight='semibold')
        ax.set_xlabel('Probability Semantic Weight')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
        
    return lda_model, corpus, dictionary

def execute_bertopic_modeling(raw_texts):
    """Runs a contextual transformer pipeline via BERTopic for dense cluster mapping."""
    print("\n--- Constructing Transformer-Based BERTopic Models ---")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(raw_texts)
    print("Transformer BERTopic pipeline executed successfully.")
    return topic_model, topic_model.get_topic_info()


# --- SECTION 4: DEEP ADVANCED ZERO-SHOT EXPERT NARRATIVE CLASSIFICATION ---

def classify_patient_narratives(unique_cleaned_texts):
    """Uses a foundation MNLI transformer to match text into structural manuscript classes."""
    print("\n--- Initiating DeBERTa Foundation Zero-Shot Categorization ---")
    classifier = pipeline("zero-shot-classification",
                          model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                          device=-1)  # Set to 0 if a local GPU instance exists
    
    class_map = {
        "Symptom Awareness": "early skin symptoms, itching, redness, and diagnostic uncertainty",
        "Clinical Care": "dermatologist visits, medical prescriptions, and clinical treatment",
        "Holistic & Advocacy": "alternative natural remedies, diet, and public patient advocacy",
        "Lived Experience": "daily emotional burden, social stigma, and peer support"
    }
    
    candidate_labels = list(class_map.values())
    results = classifier(unique_cleaned_texts[:10], candidate_labels, multi_label=True) # Trimmed sample sequence for performance safety
    
    for res in results:
        top_desc = res['labels'][0]
        short_name = [k for k, v in class_map.items() if v == top_desc][0]
        print(f"Text Snippet: {res['sequence'][:75]}...")
        print(f" -> Assigned Mapping: {short_name} (Confidence: {res['scores'][0]:.2f})")


# --- SECTION 5: NEW MANUSCRIPT MANIFEST ANALYTICS (Sentiment & Readability) ---

def compute_manuscript_sentiment(df, text_column):
    """Maps the 'Emotional Articulation Matrix' via VADER sentiment profiles."""
    print("\n--- Analyzing Emotional Articulation Profiles (VADER Metrics) ---")
    sia = SentimentIntensityAnalyzer()
    
    df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x)))
    df['compound_score'] = df['sentiment_scores'].apply(lambda d: d['compound'])
    
    # Categorize into manuscript dimensions
    df['lived_sentiment_bracket'] = pd.cut(
        df['compound_score'],
        bins=[-1.0, -0.05, 0.05, 1.0],
        labels=['Pain/Disruption Narrative', 'Neutral/Informational', 'Empowering/Coping Profile']
    )
    print("Patient Lived-Experience Sentiment Proportions:")
    print(df['lived_sentiment_bracket'].value_counts(normalize=True) * 100)
    return df

def compute_manuscript_readability(df, text_column):
    """Generates the 'Readability Profile of Related Content' (Flesch-Kincaid)."""
    print("\n--- Generating Readability Metric Matrices (Flesch-Kincaid Profiles) ---")
    
    df['flesch_kincaid_grade'] = df[text_column].apply(
        lambda x: textstat.flesch_kincaid_grade(str(x)) if len(str(x).strip()) > 10 else np.nan
    )
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=df['flesch_kincaid_grade'], color='#E27B9B', width=0.4)
    plt.title('Readability Profile of Content', fontweight='semibold')
    plt.ylabel('Equivalent Flesch-Kincaid Grade Level')
    plt.tight_layout()
    plt.show()
    
    print(f"Mean Peer-to-Peer Accessibility Grade: {df['flesch_kincaid_grade'].mean():.2f}")
    return df


# --- CONTROL BLOCK PIPELINE EXECUTION ---

def run_comprehensive_analytics_pipeline(meta1_df, meta2_df):
    """Executes the complete production-grade analysis sequence across datasets."""
    # Step 0: Warmup environment dependencies
    download_nltk_resources()
    
    # Step 1: Meta distributions and timelines
    analyze_and_plot_engagement(meta1_df, meta2_df)
    analyze_temporal_trends(meta1_df, meta2_df)
    
    # Step 2: Extract text representations and clean structural footprints
    print("\n--- Beginning Patient Text Corpus Cleaning Sequence ---")
    meta1_cleaned = meta1_df['text'].dropna().apply(light_clean_text)
    meta2_cleaned = meta2_df['text'].dropna().apply(light_clean_text)
    full_corpus_raw = pd.concat([meta1_cleaned, meta2_cleaned], ignore_index=True)
    
    # Step 3: NLP Tokenization, Lemmatization, and Classical LDA Modelling
    tokenized_corpus = advanced_nlp_preprocess(full_corpus_raw)
    lda_model, corpus, dictionary = execute_lda_modeling(tokenized_corpus, num_topics=4)
    
    # Step 4: Run Advanced Foundation & Context Models
    classify_patient_narratives(list(meta1_cleaned.unique()))
    topic_model, topic_info = execute_bertopic_modeling(list(meta2_cleaned.unique()))
    
    # Step 5: Process manuscript additions on combined data structures
    combined_text_df = pd.DataFrame({'raw_patient_narrative': full_corpus_raw})
    combined_text_df = compute_manuscript_sentiment(combined_text_df, 'raw_patient_narrative')
    combined_text_df = compute_manuscript_readability(combined_text_df, 'raw_patient_narrative')
    
    print("\n[Analysis Pipeline Successfully Completed]")
    return combined_text_df

# To execute the workflow within your environment, run:
# analytical_matrix = run_comprehensive_analytics_pipeline(meta1, meta2)
