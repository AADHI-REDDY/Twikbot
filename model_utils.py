"""
Model Utilities Module
Handles topic modeling (LDA) and machine learning classification.
"""

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import os


class TopicModeler:
    """
    LDA Topic Modeling using Gensim.
    """
    
    def __init__(self, num_topics=6, passes=10, random_state=42):
        """
        Initialize Topic Modeler.
        
        Args:
            num_topics (int): Number of topics for LDA
            passes (int): Number of passes through the corpus
            random_state (int): Random seed for reproducibility
        """
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        
    def prepare_corpus(self, documents):
        """
        Prepare corpus and dictionary from documents.
        
        Args:
            documents (list): List of tokenized documents
            
        Returns:
            tuple: (dictionary, corpus)
        """
        # Create dictionary
        self.dictionary = corpora.Dictionary(documents)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        # Create corpus (bag of words)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        
        return self.dictionary, self.corpus
    
    def train_lda(self, documents):
        """
        Train LDA model on documents.
        
        Args:
            documents (list): List of tokenized documents
            
        Returns:
            LdaModel: Trained LDA model
        """
        # Prepare corpus
        self.prepare_corpus(documents)
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=self.random_state,
            alpha='auto',
            per_word_topics=True
        )
        
        return self.lda_model
    
    def get_topic_keywords(self, num_words=10):
        """
        Get top keywords for each topic.
        
        Args:
            num_words (int): Number of top words per topic
            
        Returns:
            dict: Dictionary mapping topic IDs to keyword lists
        """
        if self.lda_model is None:
            return {}
        
        topics = {}
        for topic_id in range(self.num_topics):
            words = self.lda_model.show_topic(topic_id, num_words)
            topics[f"Topic {topic_id}"] = [word for word, _ in words]
        
        return topics
    
    def get_topic_names(self, num_words=3):
        """
        Generate descriptive names for topics based on top keywords.
        
        Args:
            num_words (int): Number of top words to use for naming
            
        Returns:
            dict: Dictionary mapping topic IDs to descriptive names
        """
        if self.lda_model is None:
            return {}
        
        topic_names = {}
        for topic_id in range(self.num_topics):
            words = self.lda_model.show_topic(topic_id, num_words)
            # Get top words and create a descriptive name
            top_words = [word for word, _ in words]
            # Capitalize first letter of each word
            name = " & ".join([word.capitalize() for word in top_words])
            topic_names[topic_id] = name
        
        return topic_names
    
    def get_document_topics(self, documents):
        """
        Get topic distribution for each document.
        
        Args:
            documents (list): List of tokenized documents
            
        Returns:
            np.ndarray: Topic distribution matrix (n_docs x n_topics)
        """
        if self.lda_model is None or self.dictionary is None:
            return np.array([])
        
        # Convert documents to corpus
        corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        
        # Get topic distributions
        topic_distributions = []
        for doc_bow in corpus:
            doc_topics = self.lda_model.get_document_topics(doc_bow, minimum_probability=0)
            # Sort by topic ID and extract probabilities
            doc_topics_sorted = sorted(doc_topics, key=lambda x: x[0])
            topic_probs = [prob for _, prob in doc_topics_sorted]
            topic_distributions.append(topic_probs)
        
        return np.array(topic_distributions)
    
    def save_model(self, model_dir='models'):
        """
        Save LDA model and dictionary.
        
        Args:
            model_dir (str): Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        if self.lda_model:
            self.lda_model.save(os.path.join(model_dir, 'lda_model.gensim'))
        
        if self.dictionary:
            self.dictionary.save(os.path.join(model_dir, 'dictionary.gensim'))
    
    def load_model(self, model_dir='models'):
        """
        Load LDA model and dictionary.
        
        Args:
            model_dir (str): Directory containing saved models
        """
        lda_path = os.path.join(model_dir, 'lda_model.gensim')
        dict_path = os.path.join(model_dir, 'dictionary.gensim')
        
        if os.path.exists(lda_path):
            self.lda_model = LdaModel.load(lda_path)
            self.num_topics = self.lda_model.num_topics
        
        if os.path.exists(dict_path):
            self.dictionary = corpora.Dictionary.load(dict_path)


class FakeAccountClassifier:
    """
    Random Forest Classifier for fake account detection.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize classifier.
        
        Args:
            n_estimators (int): Number of trees in random forest
            random_state (int): Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.metrics = {}
        
    def prepare_features(self, df, topic_distributions, behavioral_features=None):
        """
        Combine topic distributions with behavioral features.
        
        Args:
            df (pd.DataFrame): DataFrame with user data
            topic_distributions (np.ndarray): Topic distribution matrix
            behavioral_features (list): List of behavioral feature column names
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # Start with topic distributions
        X = topic_distributions
        feature_names = [f"topic_{i}" for i in range(topic_distributions.shape[1])]
        
        # Add behavioral features if available
        if behavioral_features:
            available_features = [f for f in behavioral_features if f in df.columns]
            if available_features:
                behavioral_data = df[available_features].values
                X = np.hstack([X, behavioral_data])
                feature_names.extend(available_features)
        
        # Get labels if available
        y = df['label'].values if 'label' in df.columns else None
        
        self.feature_names = feature_names
        return X, y, feature_names
    
    def train(self, X, y, test_size=0.2):
        """
        Train the classifier.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            test_size (float): Proportion of test set
            
        Returns:
            dict: Evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return self.metrics
    
    def predict(self, X):
        """
        Predict labels for new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            tuple: (predictions, probabilities)
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
    
    def get_feature_importance(self):
        """
        Get feature importance from trained model.
        
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if self.feature_names is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath='models/rf_model.joblib'):
        """
        Save trained model.
        
        Args:
            filepath (str): Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }, filepath)
    
    def load_model(self, filepath='models/rf_model.joblib'):
        """
        Load trained model.
        
        Args:
            filepath (str): Path to load model from
        """
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data['model']
            self.feature_names = data.get('feature_names')
            self.metrics = data.get('metrics', {})


def calculate_roc_curve(y_test, y_pred_proba):
    """
    Calculate ROC curve data.
    
    Args:
        y_test (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        
    Returns:
        tuple: (fpr, tpr, thresholds)
    """
    return roc_curve(y_test, y_pred_proba)
