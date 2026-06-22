Psoriasis Code: Cross-Platform Digital Ethnography of Lived Experience

This repository contains the complete multi-channel data science pipeline and reproducible code for the paper:

"What People With Psoriasis Talk About Online: Insights From a Digital Ethnography of Lived Experience"

The study implements a cross-sectional mixed-methods digital ethnography across diverse online patient ecosystems. This codebase automates programmatic data harvesting, linguistic readability assessments, and hybrid sentiment architectures across both Reddit communities and YouTube media streams.

🗺️ Unified Pipeline Architecture

The implementation bridges multi-platform text mining with qualitative clinical frameworks across four structured layers:

                     ┌──────────────────────────────┐
                     │ Multi-Channel Data Ingestion │
                     └──────────────┬───────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
[ Reddit Extraction Matrix ]                    [ YouTube Metadata Engine ]
 • r/Psoriasis, r/PsoriaticArthritis             • Video Transcript Aggregation
 • Rate-Limit Defensive Ingestion                • Engagement Metrics (Views/Likes)
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │ Linguistic Analytics Loop │
                       └────────────┬──────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
[ Readability Profiling ]                       [ Hybrid Sentiment Cascade ]
 • Flesch-Kincaid Grade Level                    • VADER (High-Speed Confidence Pass)
 • Gunning Fog / Dale-Chall                      • CardiffNLP RoBERTa (Nuance Pass)


Reddit Cohort Matrix: Streamlined ingestion across five targeted patient communities (r/Psoriasis, r/PsoriaticArthritis, r/ScalpPsoriasis, r/PsoriasisRemedies, r/DermatologyQuestions) using PRAW with inline comment tree flattening.

YouTube Video & Meta Streamer: Extracts patient education materials, engagement markers (view counts, likes), and clinical video descriptors using the YouTube Data API.

Meta: Requires special permission and a different env for data collection and analysis.

Linguistic Accessibility Engine: Evaluates text complexity, reading ease, and structural density of content via four standardized index distributions (textstat).

Hybrid Sentiment Cascade: Processes subjective patient health experiences. Evaluates expressions via VADER for standard classifications, falling back dynamically to a contextual transformer (CardiffNLP Twitter-RoBERTa-Base-Sentiment) for nuanced patient narratives.

## 🔒 Code Availability Policy

To comply with data privacy boundaries, respect social platform terms of service, and maintain strict environmental security, **this codebase is closed-source and private by default.** 

The complete, reproducible scripts are fully available to peer reviewers, journal editors, and academic researchers **upon reasonable request**. 

### How to Request Access
If you are a reviewer or researcher wishing to audit the pipeline, please contact the corresponding author at **[insert your institutional email here]** with:
1. Your name and institutional affiliation.
2. The purpose of your review or study.
3. Your GitHub username.

Once verified, you will be temporarily added as a private collaborator to this repository to inspect the data collection and text analytics pipeline architectures.

---
⚙️ Environment Setup & Installation

Prerequisites

Python 3.11.x

A Reddit Developer Application Account

A Google Cloud Console Project (with YouTube Data API v3 enabled)

Installation

Clone the repository:

git clone https://github.com/anjaliancy/psoriasis-paper.git
cd psoriasis-paper


Install the cross-platform dependency matrix:

pip install -r requirements.txt


Download required text corporate resources via NLTK:

import nltk
nltk.download('punkt')
nltk.download('stopwords')


🔐 Secure Credentials Configuration

To prevent credential leaks, API keys are completely isolated from the execution script blocks. Export your keys to your environmental shell before initializing scripts:

# Reddit API Gateways
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export REDDIT_USER_AGENT="DigitalEthnographyBot/2.0"

# Google/YouTube API Gateways
export YOUTUBE_API_KEY="your_google_cloud_youtube_api_key"


🚀 Execution Guide

The codebase is partitioned into targeted data-mining and calculation components:

1. Reddit Pipeline Execution

Harvests discussions, maps reading comprehension metrics to posts, and evaluates comment sentiment:

python reddit_analytics.py


Outputs: psoriasis-all-reddit-posts.csv, psoriasis-all-reddit-comments.csv, and reddit_ethnography_dashboard.png.

2. YouTube Meta and Content Execution

Extracts public video metrics and cross-references patient engagement indicators:

python youtube_analytics.py


Outputs: psoriasis-youtube-metadata.csv and aggregated tracking sheets.

📊 Core Dependencies & Framework Configurations

The quantitative architecture relies on the following verified library environments:

pandas (v2.1.1) & numpy (v1.26.0) — High-throughput matrix operations

praw (v7.7.1) — Reddit API wrapper

google-api-python-client (v2.100.0) — YouTube Data API orchestration

textstat (v0.7.3) — Structural text accessibility indices

vaderSentiment (v3.3.2) & transformers (v4.34.0) — Dual-stage hybrid sentiment layer

seaborn & matplotlib — Academic chart generation

🔬 Qualitative Validation Framework

The automated quantitative extractions directly feed into and complement manual clinical validation workflows:

Thematic Grounding: Computational extractions are structured to prepare text for validation under Braun and Clarke’s six-phase qualitative thematic analysis framework.

Clinical Semantic Mapping: Extracted topics, video metas, and patient sentiment expressions are mapped directly onto core Patient-Reported Outcome Measure (PROM) criteria, including the Psoriasis Symptom Diary (PSD) and Dermatology Life Quality Index (DLQI) dimensions.
