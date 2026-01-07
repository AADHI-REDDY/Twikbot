# 🐦 Fake Account and Spam Bot Detection on Twitter Using Topic Modeling

A comprehensive machine learning dashboard built with Streamlit that classifies Twitter accounts as either **Fake** or **Genuine** using Latent Dirichlet Allocation (LDA) topic modeling combined with behavioral features.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Model Architecture](#model-architecture)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## ✨ Features

### 🔍 Data Processing
- **Automated Text Cleaning**: Removes URLs, mentions, hashtags, punctuation, and numbers
- **Text Normalization**: Lowercase conversion, tokenization, stopword removal, and lemmatization
- **User Aggregation**: Combines multiple tweets per user into a single document
- **Behavioral Feature Engineering**: Calculates follower-to-friend ratios and engagement metrics

### 🧠 Machine Learning
- **Topic Modeling**: LDA (Latent Dirichlet Allocation) using Gensim
- **Classification**: Random Forest classifier for fake account detection
- **Feature Fusion**: Combines topic distributions with behavioral features
- **Model Persistence**: Save and load trained models for reuse

### 📊 Visualizations
- **Topic Word Clouds**: Visual representation of topic keywords
- **Confusion Matrix**: Heatmap showing classification performance
- **ROC Curve**: Interactive ROC curve with AUC score
- **Feature Importance**: Bar chart of most influential features
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 🔮 Prediction Interface
- **Real-time Prediction**: Test individual tweets with custom behavioral features
- **Confidence Scores**: Probability estimates for both classes
- **Interactive Input**: User-friendly interface for entering tweet data

## 📁 Project Structure

```
Twikbot/
├── app.py                  # Main Streamlit dashboard
├── preprocess.py           # Text preprocessing functions
├── model_utils.py          # Topic modeling and ML utilities
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── sample_data.csv        # Sample dataset for testing
└── models/                # Directory for saved models (created automatically)
    ├── lda_model.gensim
    ├── dictionary.gensim
    └── rf_model.joblib
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Twikbot
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Language Model (Optional)
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Download NLTK Data
The application will automatically download required NLTK data on first run, but you can also do it manually:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Upload Dataset**
   - Click "Browse files" in the sidebar
   - Upload a CSV file with the required format (see below)
   - Wait for preprocessing to complete

2. **Configure Parameters**
   - Adjust "Number of Topics" slider (default: 6)
   - Set "Test Set Size" for train-test split (default: 0.2)

3. **Train Models**
   - Click "🚀 Start Training" button
   - Wait for LDA and Random Forest training to complete
   - View results in the main dashboard

4. **Explore Results**
   - Review topic keywords and word clouds
   - Analyze classification metrics and confusion matrix
   - Examine feature importance

5. **Make Predictions**
   - Scroll to "Try Prediction" section
   - Enter tweet text and behavioral features
   - Click "🔍 Predict" to see results

6. **Save/Load Models**
   - Use "Save Models" to persist trained models
   - Use "Load Models" to reload previously trained models

## 📊 Dataset Format

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `username` | string | Twitter username/handle |
| `text` | string | Tweet text content |
| `followers_count` | integer | Number of followers |
| `friends_count` | integer | Number of friends/following |
| `statuses_count` | integer | Total number of tweets |
| `label` | integer | 0 = Genuine, 1 = Fake (optional) |

### Sample Data

```csv
username,text,followers_count,friends_count,statuses_count,label
user1,"Great product! #amazing",1500,200,500,0
user2,"Check this out http://spam.com",50,5000,10000,1
user3,"Having a wonderful day!",3000,500,800,0
```

### Supported Datasets

- **Cresci-2017**: Public dataset for social bot detection
- **Custom datasets**: Any CSV following the format above
- **Unlabeled data**: Can be used for topic modeling visualization only

## 🏗️ Model Architecture

### 1. Text Preprocessing Pipeline

```
Raw Tweet → Remove URLs/Mentions/Hashtags → Lowercase → 
Remove Punctuation/Numbers → Tokenize → Remove Stopwords → 
Lemmatize → Clean Tokens
```

### 2. Topic Modeling (LDA)

- **Algorithm**: Latent Dirichlet Allocation
- **Library**: Gensim
- **Parameters**:
  - Number of topics: Configurable (default: 6)
  - Passes: 10
  - Alpha: Auto-optimized
- **Output**: Topic distribution vector for each user

### 3. Feature Engineering

**Topic Features** (6 dimensions):
- Probability distribution across topics

**Behavioral Features** (5 dimensions):
- `followers_count`: Number of followers
- `friends_count`: Number of friends
- `statuses_count`: Number of tweets
- `follower_friend_ratio`: followers / friends
- `tweets_per_follower`: statuses / followers

**Total Feature Vector**: 11 dimensions

### 4. Classification

- **Algorithm**: Random Forest
- **Parameters**:
  - Number of estimators: 100
  - n_jobs: -1 (use all CPU cores)
- **Output**: Binary classification (0 = Genuine, 1 = Fake)

## 🎨 Screenshots

### Main Dashboard
![Dashboard Overview](https://via.placeholder.com/800x400?text=Dashboard+Overview)

### Topic Modeling Results
![Topic Word Clouds](https://via.placeholder.com/800x400?text=Topic+Word+Clouds)

### Classification Metrics
![Performance Metrics](https://via.placeholder.com/800x400?text=Performance+Metrics)

## 🛠️ Technologies Used

### Core Libraries
- **Streamlit**: Web dashboard framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Machine Learning
- **Gensim**: Topic modeling (LDA)
- **Scikit-learn**: Random Forest classifier and metrics
- **NLTK**: Natural language processing

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts
- **WordCloud**: Word cloud generation

### Utilities
- **Joblib**: Model serialization
- **spaCy**: Advanced NLP (optional)

## 📈 Performance Metrics

The model is evaluated using:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## 🔧 Configuration

### Adjustable Parameters

**In Sidebar:**
- Number of topics (2-20)
- Test set size (0.1-0.4)

**In Code:**
- LDA passes: `model_utils.py` → `TopicModeler.__init__()`
- Random Forest estimators: `model_utils.py` → `FakeAccountClassifier.__init__()`
- Preprocessing options: `preprocess.py` → `TextPreprocessor` methods

## 🐛 Troubleshooting

### Common Issues

**1. NLTK Data Not Found**
```python
import nltk
nltk.download('all')
```

**2. Memory Error with Large Datasets**
- Reduce number of topics
- Sample your dataset
- Increase system RAM

**3. Models Not Loading**
- Ensure `models/` directory exists
- Check file paths are correct
- Verify model files are not corrupted

**4. Poor Classification Performance**
- Increase number of training samples
- Adjust number of topics
- Add more behavioral features
- Balance dataset classes

## 📝 Future Enhancements

- [ ] Support for multiple languages
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Real-time Twitter API integration
- [ ] Advanced feature engineering
- [ ] Ensemble methods
- [ ] Explainable AI (SHAP, LIME)
- [ ] Deployment to cloud platforms

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Cresci et al. for the benchmark dataset
- Streamlit team for the amazing framework
- Open-source community for the libraries used

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

