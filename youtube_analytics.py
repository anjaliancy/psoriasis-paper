import pandas as pd
import isodate
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
import time
from tqdm.notebook import tqdm

# Standard NLP Libraries
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# API & Modeling Libraries
from googleapiclient.discovery import build
from googletrans import Translator
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.phrases import Phrases, Phraser  # Formally imported for medical bi-grams

# Handle conditional modeling imports cleanly
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    pass

try:
    from transformers import pipeline
except ImportError:
    pass

# --- Infrastructure Setup ---
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet', quiet=True)
    print("NLTK stopwords and wordnet data ensured.")

# --- Data Loading Functions ---
def load_psoriasis_data(video_path, comments_path, all_videos_path):
    psoriasis_videos = pd.read_csv(video_path)
    psoriasis_comments = pd.read_csv(comments_path)
    psoriasis_videos_all = pd.read_csv(all_videos_path) 
    psoriasis_videos = psoriasis_videos_all.copy() 
    return psoriasis_videos, psoriasis_comments

# --- YouTube API Interaction Functions ---
def initialize_youtube_api(yt_key):
    return build('youtube', 'v3', developerKey=yt_key)

def get_video_durations_and_shorts(youtube_service, video_ids):
    durations_data = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        video_ids_str = ','.join(chunk)
        try:
            request = youtube_service.videos().list(
                part='contentDetails',
                id=video_ids_str
            )
            response = request.execute()

            for item in response['items']:
                video_id = item['id']
                duration_iso = item['contentDetails']['duration']
                duration_timedelta = isodate.parse_duration(duration_iso)
                is_short = duration_timedelta.total_seconds() <= 60
                durations_data[video_id] = {'duration': duration_timedelta, 'is_short': is_short}
        except Exception as e:
            print(f"An error occurred during duration fetching for chunk: {e}")
            for vid_id in chunk:
                if vid_id not in durations_data:
                    durations_data[vid_id] = {'duration': pd.NaT, 'is_short': False}

    return pd.DataFrame([{'video_id': vid, **data} for vid, data in durations_data.items()])

def get_channel_details(youtube_service, video_ids):
    channel_data = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        video_ids_str = ','.join(chunk)
        try:
            request = youtube_service.videos().list(
                part='snippet',
                id=video_ids_str
            )
            response = request.execute()

            for item in response['items']:
                video_id = item['id']
                channel_title = item['snippet']['channelTitle']
                channel_id = item['snippet']['channelId']
                channel_data[video_id] = {'channel_title': channel_title, 'channel_id': channel_id}
        except Exception as e:
            print(f"An error occurred while fetching channel info: {e}")
            for vid_id in chunk:
                if vid_id not in channel_data:
                    channel_data[vid_id] = {'channel_title': None, 'channel_id': None}
                    
    return pd.DataFrame([{'video_id': vid, **data} for vid, data in channel_data.items()])

# --- Data Preprocessing & Analysis Functions ---
def preprocess_and_merge_duration(psoriasis_videos_df, durations_df):
    df = psoriasis_videos_df.copy()
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df = pd.merge(df, durations_df, on='video_id', how='left')
    df['duration_seconds'] = df['duration'].apply(lambda x: x.total_seconds() if pd.notna(x) and hasattr(x, 'total_seconds') else None)
    return df

def analyze_monthly_video_counts(df, start_date_str='2024-11-01', end_date_str='2025-11-30'):
    start_date = pd.Timestamp(start_date_str, tz='UTC')
    end_date = pd.Timestamp(end_date_str, tz='UTC')
    
    df_filtered = df.copy()
    if df_filtered['published_date'].dt.tz is None:
        df_filtered['published_date'] = df_filtered['published_date'].dt.tz_localize('UTC')
    else:
        df_filtered['published_date'] = df_filtered['published_date'].dt.tz_convert('UTC')

    filtered_videos = df_filtered[(df_filtered['published_date'] >= start_date) & (df_filtered['published_date'] <= end_date)].copy()
    filtered_videos['release_month'] = filtered_videos['published_date'].dt.to_period('M')
    
    monthly_video_counts = filtered_videos['release_month'].value_counts().sort_index().reset_index()
    monthly_video_counts.columns = ['release_month', 'video_count']
    monthly_video_counts['release_month'] = monthly_video_counts['release_month'].dt.to_timestamp()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='release_month', y='video_count', data=monthly_video_counts, marker='o')
    for x, y in zip(monthly_video_counts['release_month'], monthly_video_counts['video_count']):
        plt.text(x, y + 10, str(y), color='black', ha='center', va='bottom', fontsize=9)
    plt.title('Number of Videos Released per Month (Nov 2024 - Nov 2025)')
    plt.xlabel('Month')
    plt.ylabel('Number of Videos')
    plt.grid(True)
    plt.xticks(rotation=45)
    max_video_count = monthly_video_counts['video_count'].max() if not monthly_video_counts.empty else 100
    plt.ylim(0, max_video_count * 1.15)
    plt.yticks(np.arange(0, max_video_count * 1.15, 50))
    plt.tight_layout()
    plt.show()
    return monthly_video_counts

def add_upload_frequency_features(df):
    df['published_date'] = pd.to_datetime(df['published_date'], utc=True)
    df['day_of_week'] = df['published_date'].dt.day_name()
    df['hour_of_day'] = df['published_date'].dt.hour
    return df

def plot_upload_frequency(df):
    day_frequency = df['day_of_week'].value_counts().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    hour_frequency = df['hour_of_day'].value_counts().sort_index()

    print("\nVideo Upload Frequency by Day of the Week:")
    print(day_frequency)
    print("\nVideo Upload Frequency by Hour of the Day:")
    print(hour_frequency)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=day_frequency.index, y=day_frequency.values, palette='viridis')
    plt.title('Video Upload Frequency by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=hour_frequency.index, y=hour_frequency.values, palette='plasma')
    plt.title('Video Upload Frequency by Hour of the Day')
    plt.xlabel('Hour of the Day (24-hour format)')
    plt.ylabel('Number of Videos')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def format_video_duration(df):
    def timedelta_to_hms(td):
        if pd.isna(td) or not hasattr(td, 'total_seconds'):
            return None
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    df['formatted_duration'] = df['duration'].apply(timedelta_to_hms)
    return df

def analyze_channel_video_counts(psoriasis_videos_df, channel_info_df):
    df = pd.merge(psoriasis_videos_df, channel_info_df, on='video_id', how='left')
    channel_video_counts = df['channel_title'].value_counts().reset_index()
    channel_video_counts.columns = ['channel_title', 'video_count']
    print("Top 5 Channels by Video Count:")
    print(channel_video_counts.head(5))
    return df

def calculate_duration_ranges_and_likes(df):
    df_clean = df.dropna(subset=['duration_seconds']).copy()
    if df_clean.empty:
        print("No valid data for duration calculations.")
        return df

    max_val = df_clean['duration_seconds'].max()
    bins = [0, 60, 300, 600, max_val + 2]
    labels = ['0-1 min (Shorts)', '1-5 min', '5-10 min', '10+ min']
    df_clean['duration_range'] = pd.cut(df_clean['duration_seconds'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    average_likes_by_duration = df_clean.groupby('duration_range', observed=False)['like_count'].mean().reset_index()
    average_likes_by_duration.columns = ['Video Duration Range', 'Average Like Count']
    print("\nAverage Likes by Duration:")
    print(average_likes_by_duration)
    return df

def detect_title_languages(df, sleep_time=0.1):
    translator = Translator()
    def get_language(text):
        if pd.isna(text) or str(text).strip() == '':
            return None
        try:
            detected = translator.detect(str(text))
            return detected.lang
        except Exception:
            return None

    video_langs = []
    print("Detecting language of video titles...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing titles"):
        video_langs.append(get_language(row['title']))
        time.sleep(sleep_time)

    df['detected_title_language'] = video_langs
    print("\nDetected Language Value Counts:")
    print(df['detected_title_language'].value_counts())
    return df

def analyze_duration_correlation(df):
    df_subset = df.dropna(subset=['duration_seconds', 'like_count'])
    if df_subset.empty:
        return
    correlation = df_subset['duration_seconds'].corr(df_subset['like_count'])
    print(f"Correlation between video duration (seconds) and like count: {correlation:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration_seconds', y='like_count', data=df_subset, alpha=0.6)
    plt.title('Video Duration vs. Like Count')
    plt.xlabel('Video Duration (seconds)')
    plt.ylabel('Like Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def preprocess_text_nlp(text, stopwords_set, lemmatizer):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\([^)]*\)|\[[^]]*\]', '', text)
    
    emoji_pattern = re.compile(r"[\u2000-\u3300\ud83c-\ud83e\udbff\udfff\u2600-\u27bf]", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords_set])

def clean_english_titles(english_df):
    stopwords_english = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    df = english_df.copy()
    df['cleaned_title'] = df['title'].apply(lambda x: preprocess_text_nlp(x, stopwords_english, lemmatizer))
    return df

# --- 🚀 NEW FUNCTION: Psoriasis Adaptations for Text Data ---
def adapt_nlp_for_psoriasis(tokenized_titles):
    """
    Applies health domain modifications to text structures:
    1. Removes systemic dataset keywords that obscure deep LDA themes.
    2. Constructs medical phrase bi-grams (e.g., flare_up, light_therapy).
    """
    # System stop words common to every video in the query frame
    psoriasis_stop_words = {
        'psoriasis', 'psoratic', 'skin', 'disease', 'condition', 'chronic', 
        'video', 'youtube', 'channel', 'episode', 'vlog', 'watch', 'get'
    }
    
    # 1. Filter out broad structural noise words
    filtered_tokens = [
        [word for word in doc if word not in psoriasis_stop_words and len(word) > 2]
        for doc in tokenized_titles
    ]
    
    # 2. Extract medical phrase pairings (min_count=3 avoids single anomalous pairings)
    phrases = Phrases(filtered_tokens, min_count=3, threshold=5)
    bigram_transformer = Phraser(phrases)
    
    adapted_tokens = [bigram_transformer[doc] for doc in filtered_tokens]
    print(f"Psoriasis domain adaptation complete. Extracted localized bi-grams.")
    return adapted_tokens

def prepare_gensim_corpus(english_df_cleaned):
    tokenized_titles = [str(title).split() for title in english_df_cleaned['cleaned_title'] if pd.notna(title) and str(title).strip() != '']
    
    # Run the psoriasis specific processing adjustments
    adapted_titles = adapt_nlp_for_psoriasis(tokenized_titles)
    
    dictionary = corpora.Dictionary(adapted_titles)
    dictionary.filter_extremes(no_below=3, no_above=0.85) # Tightened constraints for clean medical clusters
    corpus = [dictionary.doc2bow(text) for text in adapted_titles]
    return dictionary, corpus, adapted_titles

def run_gensim_lda(corpus, dictionary, english_df_cleaned, n_topics=5):
    lda_model_gensim = LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary, random_state=42, passes=15)
    print("\nIdentified Domain Topics from Psoriasis-Adapted LDA:")
    for idx, topic in lda_model_gensim.print_topics(num_words=10):
        print(f"Topic {idx+1}: {topic}")

    topic_distributions = [lda_model_gensim[doc] for doc in corpus]
    def get_dominant_topic(topic_distribution):
        if not topic_distribution:
            return None
        return max(topic_distribution, key=lambda item: item[1])[0] + 1

    dominant_topics = [get_dominant_topic(td) for td in topic_distributions]
    return lda_model_gensim, pd.DataFrame({
        'video_id': english_df_cleaned['video_id'].reset_index(drop=True),
        'LDA_Topic': dominant_topics
    })

def setup_bertopic_model(titles, embedding_model_name="sentence-transformers/all-mpnet-base-v2", multilingual=False):
    if multilingual:
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorizer_model = CountVectorizer(stop_words=None, ngram_range=(1, 2))
    else:
        embedding_model = SentenceTransformer(embedding_model_name)
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    topic_model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer_model, verbose=True)
    topics, probs = topic_model.fit_transform(titles)
    return topic_model, topic_model.get_topic_info(), topics, probs

def plot_bertopic_topics(topic_info, topic_model, num_topics_to_plot=3, title_suffix=""):
    top_n_topics = topic_info[topic_info['Topic'] != -1].nlargest(num_topics_to_plot, 'Count')
    if not top_n_topics.empty:
        plt.figure(figsize=(15, 5 * len(top_n_topics)))
        for i, (_, row) in enumerate(top_n_topics.iterrows()):
            topic_id = row['Topic']
            topic_name = row['Name']
            representation = topic_model.get_topic(topic_id)
            words_df = pd.DataFrame(representation, columns=['Word', 'c-TF-IDF'])

            plt.subplot(len(top_n_topics), 1, i + 1)
            sns.barplot(x='c-TF-IDF', y='Word', data=words_df, palette='viridis')
            plt.title(f'Topic {topic_id}: {topic_name} {title_suffix}', fontsize=14)
            plt.xlabel('c-TF-IDF Score', fontsize=12)
            plt.ylabel('Word', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def analyze_high_engagement_videos(english_df):
    mean_likes = english_df['like_count'].mean()
    mean_comments = english_df['comment_count'].mean()
    
    high_engagement_videos_df = english_df[(english_df['like_count'] > mean_likes) & (english_df['comment_count'] > mean_comments)].copy()
    high_engagement_titles = list(high_engagement_videos_df['title'].unique())
    
    if high_engagement_titles:
        topic_model, topic_info, _, _ = setup_bertopic_model(high_engagement_titles)
        plot_bertopic_topics(topic_info, topic_model, num_topics_to_plot=5, title_suffix="(High Engagement Videos)")
    return high_engagement_videos_df

def run_zero_shot_classification(english_df, class_map_labels, device=-1):
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=device)
    candidate_labels = list(class_map_labels.values())
    titles = list(english_df['title'].unique())

    results = classifier(titles, candidate_labels, multi_label=True)
    classification_data = []
    for res in results:
        title = res['sequence']
        top_label_desc = res['labels'][0]
        short_name = [k for k, v in class_map_labels.items() if v == top_label_desc][0] 
        classification_data.append({'title': title, 'Category_Rich': short_name, 'Score_Rich': res['scores'][0]})

    return pd.merge(english_df, pd.DataFrame(classification_data), on='title', how='left')

def plot_category_analysis(df):
    if 'Category_Rich' not in df.columns or 'Score_Rich' not in df.columns:
        return
    
    df_plot = df.rename(columns={'Category_Rich': 'Category', 'Score_Rich': 'Score'})
    videos_per_category = df_plot['Category'].value_counts().reset_index()
    videos_per_category.columns = ['Category', 'Number of Videos']

    average_score_per_category = df_plot.groupby('Category')['Score'].mean().reset_index()
    average_score_per_category.columns = ['Category', 'Average Certainty Score']
    average_score_per_category = average_score_per_category.set_index('Category').loc[videos_per_category['Category'].tolist()].reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Category', y='Number of Videos', data=videos_per_category, palette='viridis', ax=ax1, zorder=2)
    
    ax1.set_title('Number of Videos and Average Certainty Score per Category')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Number of Videos', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # FIX: Explicit position matching sequence to clear label assignment warnings
    ax1.set_xticks(range(len(videos_per_category)))
    ax1.set_xticklabels(videos_per_category['Category'], rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    sns.lineplot(x='Category', y='Average Certainty Score', data=average_score_per_category, ax=ax2, color='red', marker='o', linewidth=2, zorder=3)
    ax2.set_ylabel('Average Certainty Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.1)

    for x_pos, y_val in enumerate(average_score_per_category['Average Certainty Score']):
        ax2.text(x_pos, y_val + 0.02, f'{y_val:.2f}', color='red', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# --- Execution Runtime Logic ---
if __name__ == "__main__":
    YT_API_KEY = 'YOUR_YOUTUBE_DATA_API_KEY' 
    PSORIRIS_VIDEOS_RAW_PATH = '/content/drive/MyDrive/Dd Research/youtube/month by month/psoriasis-1 year.csv'
    PSORIASIS_COMMENTS_PATH = '/content/drive/MyDrive/Dd Research/youtube/month by month/comments/psoriasis_comments_all-20251211.csv'
    PSORIASIS_VIDEOS_ALL_PATH = '/content/psoriasis_videos_all.csv'
    
    ZERO_SHOT_CLASS_MAP = {
        "Symptom Discovery": "Symptom discovery: itching, silver-white flakes, scalp redness...",
        "Clinical Care": "Medical journey: dermatologist consultation, biopsy, formal diagnosis...",
        "Holistic & Advocacy": "Holistic strategies and advocacy: anti-inflammatory diet...",
        "Lived Experience": "Lived burden: daily routine, moisturizing, flare-ups..."
    }

    setup_nltk()
    
    # 1. Load data
    psoriasis_videos_master, psoriasis_comments_df = load_psoriasis_data(PSORIRIS_VIDEOS_RAW_PATH, PSORIASIS_COMMENTS_PATH, PSORIASIS_VIDEOS_ALL_PATH)
    
    # 2. Extract API Info & Build Core Temporal Frame
    youtube_service = initialize_youtube_api(YT_API_KEY)
    video_ids_unique = psoriasis_videos_master['video_id'].unique().tolist()
    
    durations_df = get_video_durations_and_shorts(youtube_service, video_ids_unique)
    psoriasis_videos_master = preprocess_and_merge_duration(psoriasis_videos_master, durations_df)
    
    channel_info_df = get_channel_details(youtube_service, video_ids_unique)
    psoriasis_videos_master = analyze_channel_video_counts(psoriasis_videos_master, channel_info_df)
    
    # FIX: Mutation alignment (Applying global structural columns cleanly)
    psoriasis_videos_master = add_upload_frequency_features(psoriasis_videos_master)
    plot_upload_frequency(psoriasis_videos_master)
    
    # 3. Time Series Analytics
    _ = analyze_monthly_video_counts(psoriasis_videos_master)
    psoriasis_videos_master = format_video_duration(psoriasis_videos_master)
    _ = calculate_duration_ranges_and_likes(psoriasis_videos_master)
    
    # 4. Language Filtration & Localization
    psoriasis_videos_master = detect_title_languages(psoriasis_videos_master)
    analyze_duration_correlation(psoriasis_videos_master)
    
    # 5. English NLP Segmentation & Psoriasis-Adapted LDA Topic Modeling
    english_videos_df = psoriasis_videos_master[psoriasis_videos_master['detected_title_language'] == 'en'].copy()
    if not english_videos_df.empty:
        english_videos_df_cleaned = clean_english_titles(english_videos_df)
        
        # FIX: Gensim corpora dictionary error resolved + embedded domain adaptation
        dictionary, corpus, tokenized_titles = prepare_gensim_corpus(english_videos_df_cleaned)
        
        lda_model, lda_topic_df = run_gensim_lda(corpus, dictionary, english_df_cleaned=english_videos_df_cleaned)
        psoriasis_videos_master = pd.merge(psoriasis_videos_master, lda_topic_df, on='video_id', how='left')
        
        # 6. Advanced Transformer Topic Extraction (BERTopic)
        youtube_titles_english = list(english_videos_df['title'].unique())
        topic_model_bert, topic_info_bert, _, _ = setup_bertopic_model(youtube_titles_english)
        plot_bertopic_topics(topic_info_bert, topic_model_bert, num_topics_to_plot=3, title_suffix="(General English)")
        
        # High engagement analytics 
        _ = analyze_high_engagement_videos(english_videos_df)
        
        # 7. Deep Structural Zero-Shot Intent Pipeline
        english_videos_classified_df = run_zero_shot_classification(english_videos_df, ZERO_SHOT_CLASS_MAP, device=-1)
        plot_category_analysis(english_videos_classified_df)
        
    print("\nProcessing finished successfully.")
