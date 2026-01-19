# YouTube Comment Sentiment Analyzer 📊

## Overview

Public sentiment on YouTube videos can provide valuable insights into audience perception, engagement quality, and content reception. This project builds an **end-to-end NLP system** that extracts live YouTube comments, classifies sentiment using machine learning, and visualizes insights through an interactive dashboard.

The application processes real-world, unstructured text data and transforms it into actionable analytics such as sentiment distribution, trends, keyword analysis, and behavioral patterns.
<img width="1806" height="887" alt="image" src="https://github.com/user-attachments/assets/fd6e4328-c1ba-4f16-bec8-de745c754f82" />

Link - https://ytsentimentanalyzer.streamlit.app/

---

## Problem Statement

YouTube comments are:
- High volume
- Noisy (slang, emojis, sarcasm, negation)
- Difficult to summarize manually

The goal was to design a system that:
1. Automatically collects YouTube comments
2. Accurately classifies sentiment (Positive / Neutral / Negative)
3. Presents insights in an interpretable and interactive way

---

## Data Collection

Instead of relying on the YouTube Data API (quota-limited), comments are fetched using the lightweight `youtube-comment-downloader` library, enabling:
- URL-based extraction
- No API key requirement
- Real-time comment ingestion

Up to **2,000 comments per video** can be analyzed per request.

---

## Text Preprocessing

Raw comments undergo a custom NLP preprocessing pipeline to reduce noise and preserve sentiment signals:

- Lowercasing and whitespace normalization  
- Removal of non-alphanumeric noise  
- Stopword removal with **sentiment-aware exceptions**  
- Lemmatization  
- Explicit **negation handling** (e.g., `not good → not_good`)  

This step was critical in improving **negative sentiment recall**.

---

## Feature Engineering

Text is converted into numerical representations using:

- **TF-IDF Vectorization**
- Uni-grams, bi-grams, and tri-grams
- Frequency filtering (`min_df`, `max_df`)
- Sublinear term frequency scaling

This representation captures both individual sentiment words and short contextual phrases.

---

## Modeling Approach

Multiple models were evaluated during experimentation:

- Logistic Regression  
- Linear SVM  
- Random Forest  
- **LightGBM (final model)**  

### Final Model: LightGBM
LightGBM outperformed linear models by capturing **non-linear interactions between n-gram features**, especially in nuanced sentiment cases.

**Final performance:**
- **Accuracy:** ~87%
- **Macro F1-score:** ~0.85
- Strong improvement in negative sentiment recall

The trained model was serialized using `joblib` for reuse during inference.

---

## Deployment Pipeline

The trained model is integrated into a **Streamlit web application** that performs:

1. Comment ingestion from a YouTube URL
2. Preprocessing and sentiment prediction
3. Aggregation and visualization of results


---

## Dashboard & Visual Analytics

The interactive dashboard includes:

### Key Metrics
- Percentage of Positive, Neutral, and Negative comments
- Average sentiment score

### Visualizations
- Sentiment trend across comment index
- Keyword frequency bar charts
- Global word cloud
- Sentiment vs comment length analysis
- Most extreme positive and negative comments

These visuals help identify:
- Audience reaction patterns
- Controversial discussion points
- Engagement depth

---

## Tech Stack

- **Language:** Python  
- **NLP & ML:** scikit-learn, LightGBM, NLTK  
- **Visualization:** Plotly, Matplotlib, WordCloud  
- **Web App:** Streamlit  
- **Deployment:** GitHub + Cloud platforms  

---

## Key Learnings

- Negation-aware preprocessing significantly improves sentiment classification
- Linear models are strong baselines for text, but tree-based models can capture richer interactions
- Accuracy alone is insufficient; **macro-F1** provides a better evaluation for imbalanced sentiment classes
- UX and interpretability are as important as model performance in real-world ML systems

---

## Future Improvements

- Confidence-weighted sentiment scoring
- Multilingual sentiment support
- Sarcasm detection
- Topic modeling for thematic analysis
- Caching and performance optimization for large videos

---

## Conclusion

This project demonstrates a complete **machine learning lifecycle** — from raw data ingestion and modeling to deployment and interactive analytics. It bridges the gap between NLP research and real-world usability, showcasing how sentiment analysis can be transformed into meaningful insights for content analysis.

