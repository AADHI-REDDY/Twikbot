"""
Fake Account and Spam Bot Detection on Twitter Using Topic Modeling
A Streamlit Dashboard for detecting fake Twitter accounts using LDA and ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings

# Import custom modules
from preprocess import TextPreprocessor, aggregate_user_texts, extract_behavioral_features
from model_utils import TopicModeler, FakeAccountClassifier, calculate_roc_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Twitter Fake Account Detection",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom styling
st.markdown("""<style>
    .stMetric {font-size: 1.1rem;}
    h1 {color: #1DA1F2;}
</style>""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'topic_modeler' not in st.session_state:
    st.session_state.topic_modeler = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()


def main():
    """Main application function."""
    
    # Header
    st.title("🐦 Twitter Fake Account Detection")
    st.caption("Using Topic Modeling (LDA) and Machine Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should contain: username, text, followers_count, friends_count, statuses_count, label"
        )
        
        if uploaded_file is not None:
            load_data(uploaded_file)
        
        st.divider()
        
        # Model parameters
        st.subheader("🎛️ Model Parameters")
        num_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=20,
            value=6,
            help="Number of topics for LDA modeling"
        )
        
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data for testing"
        )
        
        st.divider()
        
        # Training button
        if st.session_state.data_loaded:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                train_models(num_topics, test_size)
        
        st.divider()
        
        # Model management - only show if models exist
        if st.session_state.model_trained:
            st.subheader("💾 Model Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Models", use_container_width=True):
                    save_models()
            
            with col2:
                if st.button("Load Models", use_container_width=True):
                    load_models()
    
    # Main content area
    if st.session_state.data_loaded:
        display_data_overview()
        
        if st.session_state.model_trained:
            display_topic_modeling_results()
            display_classification_results()
            display_prediction_interface()
    else:
        display_welcome_screen()


def display_welcome_screen():
    """Display welcome screen with instructions."""
    st.info("👋 Welcome! Upload a Twitter dataset to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Required Columns
        - `username` - Twitter username
        - `text` - Tweet text
        - `followers_count` - Follower count
        - `friends_count` - Following count
        - `statuses_count` - Tweet count
        - `label` - 0=Genuine, 1=Fake (optional)
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Features
        - Text preprocessing & cleaning
        - LDA topic modeling
        - Random Forest classification
        - Interactive visualizations
        - Real-time predictions
        """)


def load_data(uploaded_file):
    """Load and preprocess uploaded data."""
    try:
        with st.spinner("📥 Loading and preprocessing data..."):
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['username', 'text']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Add default values for missing optional columns
            if 'followers_count' not in df.columns:
                df['followers_count'] = 0
            if 'friends_count' not in df.columns:
                df['friends_count'] = 0
            if 'statuses_count' not in df.columns:
                df['statuses_count'] = 0
            if 'label' not in df.columns:
                df['label'] = -1  # Unknown label
            
            # Preprocess text
            preprocessor = st.session_state.preprocessor
            df = preprocessor.preprocess_dataframe(df, text_column='text')
            
            # Aggregate by user
            df = aggregate_user_texts(df, user_column='username', text_column='cleaned_text')
            
            # Extract behavioral features
            df = extract_behavioral_features(df)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success(f"✅ Data loaded successfully! {len(df)} users processed.")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")


def display_data_overview():
    """Display overview of loaded data."""
    st.header("📊 Data Overview")
    
    df = st.session_state.df
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(df))
    
    with col2:
        if 'label' in df.columns and df['label'].max() >= 0:
            genuine_count = (df['label'] == 0).sum()
            st.metric("Genuine Accounts", genuine_count)
    
    with col3:
        if 'label' in df.columns and df['label'].max() >= 0:
            fake_count = (df['label'] == 1).sum()
            st.metric("Fake Accounts", fake_count)
    
    with col4:
        avg_followers = int(df['followers_count'].mean())
        st.metric("Avg Followers", f"{avg_followers:,}")
    
    # Data preview
    with st.expander("🔍 View Dataset Sample"):
        display_cols = ['username', 'cleaned_text', 'followers_count', 'friends_count', 'statuses_count']
        if 'label' in df.columns:
            display_cols.append('label')
        
        st.dataframe(df[display_cols].head(10), use_container_width=True)


def train_models(num_topics, test_size):
    """Train topic modeling and classification models."""
    try:
        df = st.session_state.df
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare documents (tokenized text)
        documents = df['cleaned_text'].apply(lambda x: x.split()).tolist()
        
        # Train topic model
        status_text.text("📚 Training LDA topic model...")
        progress_bar.progress(20)
        topic_modeler = TopicModeler(num_topics=num_topics)
        topic_modeler.train_lda(documents)
        progress_bar.progress(50)
        
        # Get topic distributions
        topic_distributions = topic_modeler.get_document_topics(documents)
        st.session_state.topic_modeler = topic_modeler
        
        # Train classifier if labels are available
        if 'label' in df.columns and df['label'].max() >= 0:
            status_text.text("🤖 Training Random Forest classifier...")
            progress_bar.progress(70)
            
            classifier = FakeAccountClassifier()
            
            # Prepare features
            behavioral_features = [
                'followers_count', 'friends_count', 'statuses_count',
                'follower_friend_ratio', 'tweets_per_follower'
            ]
            
            X, y, feature_names = classifier.prepare_features(
                df, topic_distributions, behavioral_features
            )
            
            # Train
            metrics = classifier.train(X, y, test_size=test_size)
            progress_bar.progress(100)
            
            # Store classifier
            st.session_state.classifier = classifier
            st.session_state.model_trained = True
            
            status_text.empty()
            progress_bar.empty()
            st.success("✅ Training complete!")
        else:
            st.session_state.model_trained = True
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            st.warning("⚠️ No labels found. Only topic modeling available.")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def display_topic_modeling_results():
    """Display topic modeling results."""
    st.header("📚 Topic Modeling Results")
    
    topic_modeler = st.session_state.topic_modeler
    
    if topic_modeler is None:
        return
    
    # Get topic keywords and descriptive names
    topics = topic_modeler.get_topic_keywords(num_words=10)
    topic_names = topic_modeler.get_topic_names(num_words=3)
    
    # Display topics with descriptive names
    st.subheader("🔑 Discovered Topics")
    
    cols = st.columns(3)
    for idx, (topic_id, keywords) in enumerate(topics.items()):
        topic_num = int(topic_id.split()[1])  # Extract number from "Topic 0"
        descriptive_name = topic_names.get(topic_num, f"Topic {topic_num}")
        
        with cols[idx % 3]:
            st.markdown(f"**{descriptive_name}**")
            st.caption(f"Top keywords:")
            st.write(", ".join(keywords[:5]))  # Show top 5 keywords
    
    st.divider()
    
    # Word clouds with descriptive names
    st.subheader("☁️ Topic Word Clouds")
    
    num_topics = len(topics)
    cols_per_row = 3
    num_rows = (num_topics + cols_per_row - 1) // cols_per_row
    
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            topic_idx = row * cols_per_row + col_idx
            if topic_idx < num_topics:
                topic_id = f"Topic {topic_idx}"
                keywords = topics[topic_id]
                descriptive_name = topic_names.get(topic_idx, f"Topic {topic_idx}")
                
                with cols[col_idx]:
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=400,
                        height=300,
                        background_color='white',
                        colormap='viridis'
                    ).generate(' '.join(keywords))
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(descriptive_name, fontsize=11, fontweight='bold')
                    st.pyplot(fig)
                    plt.close()
    
    # Topic Distribution Analysis
    st.divider()
    st.subheader("📊 Topic Distribution Analysis")
    
    df = st.session_state.df
    if df is not None and 'cleaned_text' in df.columns:
        # Get topic distributions for all documents
        documents = df['cleaned_text'].apply(lambda x: x.split()).tolist()
        topic_distributions = topic_modeler.get_document_topics(documents)
        
        if len(topic_distributions) > 0:
            # Calculate average topic distribution
            avg_topic_dist = topic_distributions.mean(axis=0)
            
            # Get descriptive topic names for charts
            descriptive_topic_names = [topic_names.get(i, f"Topic {i}") for i in range(len(avg_topic_dist))]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of average topic distribution
                fig = px.bar(
                    x=descriptive_topic_names,
                    y=avg_topic_dist,
                    title='Average Topic Distribution Across All Documents',
                    labels={'x': 'Topic', 'y': 'Average Probability'},
                    color=avg_topic_dist,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart of topic prevalence
                fig = go.Figure(data=[go.Pie(
                    labels=descriptive_topic_names,
                    values=avg_topic_dist,
                    hole=0.3,
                    textinfo='label+percent'
                )])
                fig.update_layout(
                    title='Topic Prevalence Distribution',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of topic distributions (sample)
            st.subheader("🔥 Topic Distribution Heatmap (Sample)")
            sample_size = min(20, len(topic_distributions))
            sample_dist = topic_distributions[:sample_size]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(
                sample_dist.T,
                cmap='YlOrRd',
                xticklabels=[f"Doc {i+1}" for i in range(sample_size)],
                yticklabels=descriptive_topic_names,
                cbar_kws={'label': 'Probability'},
                ax=ax
            )
            ax.set_xlabel('Documents (Sample)')
            ax.set_ylabel('Topics')
            ax.set_title(f'Topic Distribution Heatmap (First {sample_size} Documents)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


def display_classification_results():
    """Display classification results and metrics."""
    classifier = st.session_state.classifier
    
    if classifier is None or not classifier.metrics:
        return
    
    st.header("🎯 Classification Results")
    
    metrics = classifier.metrics
    
    # Metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.3f}")
    with col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        cm = metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Genuine', 'Fake'],
            yticklabels=['Genuine', 'Fake'],
            ax=ax
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # ROC Curve
        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = calculate_roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {metrics["roc_auc"]:.3f})',
            line=dict(color='#1DA1F2', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=500,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.divider()
    st.subheader("🔍 Feature Importance")
    
    importance_df = classifier.get_feature_importance()
    
    if not importance_df.empty:
        fig = px.bar(
            importance_df.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Most Important Features',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional visualizations
    st.divider()
    
    # Precision-Recall Curve and Metrics Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision-Recall Curve
        st.subheader("📉 Precision-Recall Curve")
        precision_vals, recall_vals, _ = precision_recall_curve(metrics['y_test'], metrics['y_pred_proba'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_vals, y=precision_vals,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=500,
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metrics Comparison Bar Chart
        st.subheader("📊 Performance Metrics Comparison")
        metrics_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['roc_auc']
            ]
        })
        
        fig = px.bar(
            metrics_data,
            x='Metric',
            y='Score',
            title='All Performance Metrics',
            color='Score',
            color_continuous_scale='Greens',
            text='Score'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            height=400,
            yaxis=dict(range=[0, 1.1]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Class Distribution
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction Distribution
        st.subheader("🎯 Prediction Distribution")
        pred_counts = pd.Series(metrics['y_pred']).value_counts().sort_index()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Genuine', 'Fake'],
            values=pred_counts.values,
            hole=0.4,
            marker=dict(colors=['#4CAF50', '#F44336']),
            textinfo='label+percent+value'
        )])
        fig.update_layout(
            title='Predicted Class Distribution',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # True Distribution
        st.subheader("✅ True Distribution")
        true_counts = pd.Series(metrics['y_test']).value_counts().sort_index()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Genuine', 'Fake'],
            values=true_counts.values,
            hole=0.4,
            marker=dict(colors=['#4CAF50', '#F44336']),
            textinfo='label+percent+value'
        )])
        fig.update_layout(
            title='True Class Distribution',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.divider()
    st.subheader("📋 Detailed Classification Report")
    
    report = classification_report(
        metrics['y_test'], 
        metrics['y_pred'],
        target_names=['Genuine', 'Fake'],
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(3)
    
    # Style the dataframe
    st.dataframe(
        report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
        use_container_width=True
    )


def display_prediction_interface():
    """Display interface for making predictions on new tweets."""
    st.header("🔮 Try Prediction")
    
    st.markdown("Enter a tweet text to predict if the account is **Fake** or **Genuine**.")
    
    # Helper info
    with st.expander("💡 Tips for Accurate Predictions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Fake Account Patterns:**
            - Low followers (50-200)
            - High friends (5000-15000)
            - High statuses (10000-40000)
            - Spam keywords: "Click", "Buy", "Free", "Win"
            """)
        with col2:
            st.markdown("""
            **Genuine Account Patterns:**
            - Balanced follower/friend ratio
            - Moderate activity (100-2000 tweets)
            - Natural language
            - Personal/professional content
            """)
    
    # Input text
    tweet_text = st.text_area(
        "Tweet Text",
        placeholder="Enter tweet text here...",
        height=100
    )
    
    # Spam keyword detection
    spam_keywords = ['click', 'buy', 'free', 'win', 'prize', 'gift', 'follow', 'urgent', 'limited', 'offer', 'now', 'claim']
    if tweet_text:
        text_lower = tweet_text.lower()
        found_spam_words = [word for word in spam_keywords if word in text_lower]
        if found_spam_words:
            st.warning(f"⚠️ Spam indicators detected: {', '.join(found_spam_words)}")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        followers = st.number_input("Followers Count", min_value=0, value=1000, help="Fake accounts usually have 50-200")
    with col2:
        friends = st.number_input("Friends Count", min_value=0, value=500, help="Fake accounts usually have 5000-15000")
    with col3:
        statuses = st.number_input("Statuses Count", min_value=0, value=100, help="Fake accounts usually have 10000-40000")
    
    # Quick presets
    st.markdown("**Quick Presets:**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("🤖 Typical Fake Account", use_container_width=True):
            st.session_state.preset_followers = 80
            st.session_state.preset_friends = 8000
            st.session_state.preset_statuses = 15000
            st.rerun()
    
    with preset_col2:
        if st.button("✅ Typical Genuine Account", use_container_width=True):
            st.session_state.preset_followers = 2000
            st.session_state.preset_friends = 400
            st.session_state.preset_statuses = 500
            st.rerun()
    
    with preset_col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.preset_followers = 1000
            st.session_state.preset_friends = 500
            st.session_state.preset_statuses = 100
            st.rerun()
    
    # Apply presets if they exist
    if 'preset_followers' in st.session_state:
        followers = st.session_state.preset_followers
    if 'preset_friends' in st.session_state:
        friends = st.session_state.preset_friends
    if 'preset_statuses' in st.session_state:
        statuses = st.session_state.preset_statuses
    
    if st.button("🔍 Predict", type="primary"):
        if tweet_text.strip():
            make_prediction(tweet_text, followers, friends, statuses)
        else:
            st.warning("⚠️ Please enter some tweet text.")


def make_prediction(tweet_text, followers, friends, statuses):
    """Make prediction for a single tweet."""
    try:
        classifier = st.session_state.classifier
        topic_modeler = st.session_state.topic_modeler
        preprocessor = st.session_state.preprocessor
        
        if classifier is None or topic_modeler is None:
            st.error("❌ Models not trained. Please train models first.")
            return
        
        # Preprocess text
        tokens = preprocessor.preprocess_text(tweet_text)
        
        if not tokens:
            st.warning("⚠️ No valid text after preprocessing.")
            return
        
        # Get topic distribution
        topic_dist = topic_modeler.get_document_topics([tokens])
        
        # Calculate behavioral features
        follower_friend_ratio = followers / max(friends, 1)
        tweets_per_follower = statuses / max(followers, 1)
        
        # Combine features
        behavioral_features = np.array([[
            followers, friends, statuses,
            follower_friend_ratio, tweets_per_follower
        ]])
        
        X = np.hstack([topic_dist, behavioral_features])
        
        # Predict
        prediction, probabilities = classifier.predict(X)
        
        # Display result
        st.divider()
        
        if prediction[0] == 1:
            st.error("🚨 **Prediction: FAKE ACCOUNT**")
            confidence = probabilities[0][1] * 100
        else:
            st.success("✅ **Prediction: GENUINE ACCOUNT**")
            confidence = probabilities[0][0] * 100
        
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Genuine Probability", f"{probabilities[0][0]:.3f}")
        with col2:
            st.metric("Fake Probability", f"{probabilities[0][1]:.3f}")
        
        # Analysis of features
        st.divider()
        st.subheader("📊 Feature Analysis")
        
        # Behavioral analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Behavioral Signals:**")
            
            # Follower/Friend ratio analysis
            if follower_friend_ratio < 0.1:
                st.error(f"🔴 Very low follower/friend ratio: {follower_friend_ratio:.3f} (Fake indicator)")
            elif follower_friend_ratio < 0.5:
                st.warning(f"🟡 Low follower/friend ratio: {follower_friend_ratio:.3f}")
            else:
                st.success(f"🟢 Good follower/friend ratio: {follower_friend_ratio:.3f}")
            
            # Friends count analysis
            if friends > 5000:
                st.error(f"🔴 Very high friends count: {friends:,} (Fake indicator)")
            elif friends > 2000:
                st.warning(f"🟡 High friends count: {friends:,}")
            else:
                st.success(f"🟢 Normal friends count: {friends:,}")
            
            # Statuses analysis
            if statuses > 10000:
                st.error(f"🔴 Very high tweet count: {statuses:,} (Fake indicator)")
            elif statuses > 5000:
                st.warning(f"🟡 High tweet count: {statuses:,}")
            else:
                st.success(f"🟢 Normal tweet count: {statuses:,}")
        
        with col2:
            st.markdown("**Content Signals:**")
            
            # Spam keyword check
            spam_keywords = ['click', 'buy', 'free', 'win', 'prize', 'gift', 'follow', 'urgent', 'limited', 'offer', 'now', 'claim']
            text_lower = tweet_text.lower()
            found_spam = [word for word in spam_keywords if word in text_lower]
            
            if len(found_spam) >= 3:
                st.error(f"🔴 Multiple spam keywords: {', '.join(found_spam)}")
            elif len(found_spam) > 0:
                st.warning(f"🟡 Spam keywords found: {', '.join(found_spam)}")
            else:
                st.success("🟢 No obvious spam keywords")
            
            # URL check
            if 'http://' in text_lower or 'https://' in text_lower or 'bit.ly' in text_lower:
                st.warning("🟡 Contains URL (common in spam)")
            else:
                st.success("🟢 No URLs detected")
            
            # Length check
            if len(tweet_text) < 20:
                st.warning("🟡 Very short text")
            else:
                st.success(f"🟢 Reasonable length ({len(tweet_text)} chars)")
        
        # Overall assessment
        st.divider()
        st.markdown("**💡 Why this prediction?**")
        
        # Count red flags
        red_flags = 0
        if follower_friend_ratio < 0.1:
            red_flags += 1
        if friends > 5000:
            red_flags += 1
        if statuses > 10000:
            red_flags += 1
        if len(found_spam) >= 2:
            red_flags += 1
        
        if red_flags >= 3:
            st.error(f"⚠️ **{red_flags} red flags detected** - Strong fake account indicators present")
        elif red_flags >= 2:
            st.warning(f"⚠️ **{red_flags} red flags detected** - Some fake account indicators present")
        elif red_flags == 1:
            st.info(f"ℹ️ **{red_flags} red flag detected** - Minor concern")
        else:
            st.success("✅ **No major red flags** - Appears to be genuine behavior")
        
        # Explanation
        if prediction[0] == 0 and (len(found_spam) > 0 or red_flags > 0):
            st.info("ℹ️ **Note:** The model predicted GENUINE because the behavioral features (followers, friends, statuses) look normal, even though the text contains spam indicators. For better fake detection, use realistic fake account numbers (low followers, high friends/statuses).")
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")


def save_models():
    """Save trained models to disk."""
    try:
        saved = []
        if st.session_state.topic_modeler:
            st.session_state.topic_modeler.save_model('models')
            saved.append("Topic model")
        
        if st.session_state.classifier:
            st.session_state.classifier.save_model('models/rf_model.joblib')
            saved.append("Classifier")
        
        if saved:
            st.success(f"✅ Saved: {', '.join(saved)}")
        else:
            st.warning("⚠️ No models to save")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def load_models():
    """Load trained models from disk."""
    try:
        loaded = []
        
        if os.path.exists('models/lda_model.gensim'):
            topic_modeler = TopicModeler()
            topic_modeler.load_model('models')
            st.session_state.topic_modeler = topic_modeler
            loaded.append("Topic model")
        
        if os.path.exists('models/rf_model.joblib'):
            classifier = FakeAccountClassifier()
            classifier.load_model('models/rf_model.joblib')
            st.session_state.classifier = classifier
            st.session_state.model_trained = True
            loaded.append("Classifier")
        
        if loaded:
            st.success(f"✅ Loaded: {', '.join(loaded)}")
        else:
            st.warning("⚠️ No saved models found")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
