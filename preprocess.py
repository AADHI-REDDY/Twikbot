"""
Text Preprocessing Module for Twitter Data
Handles cleaning, tokenization, and normalization of tweet text.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for Twitter data.
    """
    
    def __init__(self):
        """Initialize preprocessor with stopwords and lemmatizer."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def remove_urls(self, text):
        """Remove URLs from text."""
        if pd.isna(text):
            return ""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', str(text))
    
    def remove_mentions(self, text):
        """Remove @mentions from text."""
        if pd.isna(text):
            return ""
        mention_pattern = re.compile(r'@\w+')
        return mention_pattern.sub('', str(text))
    
    def remove_hashtags(self, text):
        """Remove hashtags from text."""
        if pd.isna(text):
            return ""
        hashtag_pattern = re.compile(r'#\w+')
        return hashtag_pattern.sub('', str(text))
    
    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        if pd.isna(text):
            return ""
        return str(text).translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text):
        """Remove numbers from text."""
        if pd.isna(text):
            return ""
        return re.sub(r'\d+', '', str(text))
    
    def to_lowercase(self, text):
        """Convert text to lowercase."""
        if pd.isna(text):
            return ""
        return str(text).lower()
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace from text."""
        if pd.isna(text):
            return ""
        return ' '.join(str(text).split())
    
    def tokenize(self, text):
        """Tokenize text into words."""
        if pd.isna(text) or text == "":
            return []
        return word_tokenize(str(text))
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        # Step 1: Remove URLs, mentions, hashtags
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        
        # Step 2: Convert to lowercase
        text = self.to_lowercase(text)
        
        # Step 3: Remove punctuation and numbers
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        
        # Step 4: Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Step 5: Tokenize
        tokens = self.tokenize(text)
        
        # Step 6: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 7: Lemmatize
        tokens = self.lemmatize(tokens)
        
        return tokens
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preprocess text column in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            
        Returns:
            pd.DataFrame: DataFrame with additional preprocessed columns
        """
        df = df.copy()
        
        # Apply preprocessing
        df['tokens'] = df[text_column].apply(self.preprocess_text)
        
        # Create cleaned text column (join tokens back)
        df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        # Filter out empty texts
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        
        return df


def aggregate_user_texts(df, user_column='username', text_column='cleaned_text'):
    """
    Aggregate all tweets per user into a single document.
    
    Args:
        df (pd.DataFrame): DataFrame with user tweets
        user_column (str): Column name for username
        text_column (str): Column name for cleaned text
        
    Returns:
        pd.DataFrame: Aggregated DataFrame with one row per user
    """
    # Group by user and aggregate
    user_df = df.groupby(user_column).agg({
        text_column: lambda x: ' '.join(x),
        'followers_count': 'first' if 'followers_count' in df.columns else lambda x: 0,
        'friends_count': 'first' if 'friends_count' in df.columns else lambda x: 0,
        'statuses_count': 'first' if 'statuses_count' in df.columns else lambda x: 0,
        'label': 'first' if 'label' in df.columns else lambda x: -1
    }).reset_index()
    
    return user_df


def extract_behavioral_features(df):
    """
    Extract behavioral features from user data.
    
    Args:
        df (pd.DataFrame): DataFrame with user information
        
    Returns:
        pd.DataFrame: DataFrame with additional behavioral features
    """
    df = df.copy()
    
    # Calculate derived features
    if 'followers_count' in df.columns and 'friends_count' in df.columns:
        # Follower-to-friend ratio (handle division by zero)
        df['follower_friend_ratio'] = df.apply(
            lambda row: row['followers_count'] / max(row['friends_count'], 1), 
            axis=1
        )
    
    if 'statuses_count' in df.columns and 'followers_count' in df.columns:
        # Tweets per follower
        df['tweets_per_follower'] = df.apply(
            lambda row: row['statuses_count'] / max(row['followers_count'], 1),
            axis=1
        )
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    return df
