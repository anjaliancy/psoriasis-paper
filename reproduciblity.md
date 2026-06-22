# Supplementary Methods: Quantitative & Qualitative Analysis Reproducibility Framework

## Statistical & Computational Environment

Data collected was between November 2024 and November 2025.

All quantitative analyses were performed using Python version 3.11.5. Data cleaning, preprocessing, engagement analysis, frequency extraction, sentiment scoring, topic modelling, and readability assessment were implemented using reproducible scripts provided in Supplementary File 1. Analyses were conducted between Dec 2025, and Jan 2026. 

The software environment was strictly controlled to ensure deterministic outputs for all stochastic models (including Latent Dirichlet Allocation and BERTopic). Package versions included:
* **pandas**: 2.1.1
* **NumPy**: 1.26.0
* **scikit-learn**: 1.3.1
* **NLTK**: 3.8.1
* **vaderSentiment**: 3.3.2
* **gensim**: 4.3.2
* **BERTopic**: 0.15.0
* **sentence-transformers**: 2.2.2
* **textstat**: 0.7.3
* **transformers**: 4.34.0

## Qualitative Coding & Analysis Framework

Qualitative thematic coding was performed manually using Microsoft Excel and Python-extracted data frames, following Braun and Clarke’s six-phase thematic analysis framework to identify, analyze, and report patterns (themes) within the digital patient narratives. 

Patient-reported outcome semantic mapping was conducted against predefined Patient-Reported Outcome Measure (PROM) domains derived from established dermatological frameworks, specifically focusing on the Psoriasis Symptom Diary (PSD) and the Dermatology Life Quality Index (DLQI) core dimensions (Symptoms, Daily Activities, Leisure, Work/School, Personal Relationships, and Treatment Effects) to map patient-expressed online concerns directly to validated clinical outcome endpoints.

