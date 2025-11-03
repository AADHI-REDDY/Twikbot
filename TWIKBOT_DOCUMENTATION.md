# Twitter Fake Account and Spam Bot Detection Using Topic Modeling

## Document Version
- Version: 1.0
- Date: 2025-10-31
- Project: Twikbot (Streamlit-based dashboard)

---

## Executive Summary
This project detects fake (spam/bot) Twitter accounts by combining text-based topic modeling with behavioral account features. The solution provides a fully interactive Streamlit dashboard for data upload, preprocessing, topic discovery (LDA), model training (Random Forest), evaluation (metrics and plots), and a real-time prediction interface. It is designed for analysts who need explainable results, operational teams who require a simple UI, and data scientists who want modular, extensible code.

---

## Table of Contents
1. Introduction
2. System Overview
3. Data Requirements
4. Preprocessing Pipeline
5. Topic Modeling (LDA)
6. Feature Engineering
7. Classification Model
8. Evaluation and Visualizations
9. Application Workflow
10. Usage Instructions
11. Results Snapshot
12. Limitations and Risks
13. Future Enhancements
14. Installation and Setup
15. Troubleshooting
16. Project Structure
17. License

---

## 1. Introduction
- Objective: Identify fake (spam/bot) Twitter accounts using a hybrid approach that leverages both textual patterns and behavioral signals.
- Approach: Latent Dirichlet Allocation (LDA) extracts latent topics from tweet text; engineered behavioral features strengthen the classifier’s ability to distinguish bots from genuine users.
- Deliverable: A Streamlit dashboard with training, evaluation, and prediction capabilities.

---

## 2. System Overview
- UI: Streamlit web app (single-page, modular sections)
- Core Libraries: pandas, numpy, gensim (LDA), scikit-learn (RandomForest, metrics), nltk, matplotlib, seaborn, plotly, wordcloud, joblib
- Persistence: Models can be saved/loaded from the local models/ directory
- Explainability: Topic keywords/word clouds, feature importances, PR/ROC curves, confusion matrix, and a per-prediction feature analysis panel

High-level data flow:
1. CSV Upload → 2. Text Cleaning → 3. Tokenization & Lemmatization → 4. LDA Topic Modeling → 5. Topic Distributions + Behavioral Features → 6. Train/Test Split → 7. Random Forest Training → 8. Metrics & Visuals → 9. Prediction Interface

---

## 3. Data Requirements
Expected CSV columns:
- username (str)
- text (str): tweet content
- followers_count (int)
- friends_count (int)
- statuses_count (int)
- label (int, optional): 0 = Genuine, 1 = Fake (used for supervised training/evaluation)

Notes:
- If label is absent, the app still runs topic modeling but skips supervised metrics.
- Sample dataset: sample_data.csv included for quick testing.

---

## 4. Preprocessing Pipeline
Module: preprocess.py
Steps:
- Lowercasing, URL/mention/hashtag removal
- Punctuation/number stripping
- Tokenization
- Stopword removal (NLTK)
- Lemmatization (WordNet)
Output:
- cleaned_text column with normalized, space-separated tokens

Quality considerations:
- Unicode handling is robust; common Windows console encoding warnings are cosmetic.

---

## 5. Topic Modeling (LDA)
Module: model_utils.py (class: TopicModeler)
- Dictionary creation and corpus generation
- Gensim LDA with alpha='auto', per_word_topics=True
- Configurable number of topics (2–20)
- Topic keyword extraction
- Automatic topic naming: descriptive titles generated from top keywords (e.g., “Spam & Click & Buy”) for intuitive charts and tables
Outputs:
- Topic distribution matrix per document
- Keyword lists and word clouds for each topic

---

## 6. Feature Engineering
Behavioral features (per account):
- followers_count, friends_count, statuses_count
- follower_friend_ratio = followers_count / max(friends_count, 1)
- tweets_per_follower = statuses_count / max(followers_count, 1)

Combined features:
- [Topic distribution vector] + [Behavioral features] → model input

Rationale:
- Behavioral signals (e.g., very low ratio + very high friends) are strong indicators of spam/bot behavior.

---

## 7. Classification Model
Classifier: RandomForestClassifier (scikit-learn)
- Trained on topic + behavioral features
- Split: configurable test_size (default 0.2)
- Saved/loaded with joblib
Metrics reported:
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix
- Precision–Recall and ROC curves
- Classification report (per-class precision/recall/F1, support)
- Feature importance (top features shown)

---

## 8. Evaluation and Visualizations
Always-visible visuals:
- Topic keywords grid and word clouds
- Average topic distribution bar, topic prevalence pie
- Confusion matrix and ROC curve
- Feature importance bar (top 15)
- Precision–Recall curve and metrics comparison bar
- Predicted vs True class distribution pies
- Classification report (styled table)

Interpretation aids:
- Descriptive topic names
- Hover tooltips in Plotly charts

---

## 9. Application Workflow
1. Upload CSV and validate required columns
2. Preprocess text → add cleaned_text
3. Train LDA with chosen number of topics
4. Compute topic distributions for all documents
5. If labels exist: train Random Forest, compute metrics, render visuals
6. Use the prediction interface to test tweets with account numbers; view probabilities and feature analysis
7. Optionally save/load models

---

## 10. Usage Instructions
CLI (from project root):
- First-time setup (downloads NLTK data, installs dependencies):
  - Windows: `python setup.py`
- Run the app:
  - `streamlit run app.py`
- Or use convenience script:
  - `run.bat`

VS Code:
- Open folder → Terminal → (optional) create and select `.venv` → `pip install -r requirements.txt` → `streamlit run app.py`
- Optional launch.json provided in instructions to bind F5 to Streamlit

---

## 11. Results Snapshot
- With balanced, labeled data, the model typically achieves strong ROC-AUC (dataset-dependent)
- Feature importance often highlights follower_friend_ratio, friends_count, and key spam-topic probabilities
- Prediction interface explains outcomes with behavioral signals, content keyword checks, and red flag counter

---

## 12. Limitations and Risks
- Topic models are sensitive to preprocessing quality and number of topics
- If labels are noisy or class-imbalanced, metrics may be misleading
- Behavioral heuristics vary by domain/time; thresholds are indicative, not absolute
- Short or multilingual texts may degrade topic quality without tailored preprocessing

Mitigations:
- Tune number of topics (5–8 often works well)
- Curate/augment labeled data
- Add domain-specific stopwords
- Calibrate probability thresholds if deploying

---

## 13. Future Enhancements
- Allow manual topic renaming and per-topic notes
- Alternative models (e.g., LightGBM, XGBoost), and stacking
- Use phrase detection (bigrams/trigrams) to improve topics
- Integrate modern language models (e.g., transformer embeddings) for text features
- Model monitoring and drift detection for production use

---

## 14. Installation and Setup
Requirements: Python 3.8+

Option A (one command):
- `python setup.py`

Option B (manual):
1. `python -m venv .venv`
2. `.\.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python -c "import nltk; [nltk.download(x) for x in ['punkt','stopwords','wordnet','omw-1.4']]"`
5. `streamlit run app.py`

---

## 15. Troubleshooting
- Port in use → `streamlit run app.py --server.port 8502`
- Package build error → install without strict pinning or use wheels (`pip install streamlit pandas numpy gensim scikit-learn nltk matplotlib seaborn wordcloud joblib plotly`)
- Unicode console warnings on Windows → harmless; app unaffected
- Empty predictions → ensure text is non-empty after preprocessing and numeric inputs are realistic

---

## 16. Project Structure
```
Twikbot/
├── app.py
├── preprocess.py
├── model_utils.py
├── sample_data.csv
├── models/
├── requirements.txt
├── setup.py
├── run.bat
├── streamlit_config.toml
├── README.md
├── QUICKSTART.md
├── INSTALLATION_GUIDE.md
├── PROJECT_OVERVIEW.md
└── LICENSE
```

---

## 17. License
- MIT License (see LICENSE file)

---

## Appendix A. Prediction Interface Guide
- Inputs: tweet text, followers_count, friends_count, statuses_count
- Heuristics for testing:
  - Fake: followers≈50–200, friends≈5,000–15,000, statuses≈10,000–40,000
  - Genuine: balanced ratio (>0.5), friends<2,000, statuses<5,000
- Output: class label, confidence, class probabilities, behavioral/content analysis, red flag counter

## Appendix B. Reproducibility
- Set the same random_state in TopicModeler and classifier for reproducible splits
- Document versions in requirements.txt or freeze with `pip freeze > requirements_lock.txt`
