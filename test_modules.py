"""
Test script to verify all modules are working correctly
Run this before starting the Streamlit app
"""

import sys


def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    modules = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'gensim': 'Gensim',
        'sklearn': 'Scikit-learn',
        'nltk': 'NLTK',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'wordcloud': 'WordCloud',
        'joblib': 'Joblib',
        'plotly': 'Plotly'
    }
    
    failed = []
    
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_custom_modules():
    """Test if custom modules can be imported."""
    print("\n🔍 Testing custom modules...")
    
    modules = {
        'preprocess': 'Text Preprocessor',
        'model_utils': 'Model Utilities'
    }
    
    failed = []
    
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_nltk_data():
    """Test if NLTK data is available."""
    print("\n🔍 Testing NLTK data...")
    
    import nltk
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    failed = []
    
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
            print(f"  ✅ {dataset}")
        except LookupError:
            print(f"  ❌ {dataset} not found")
            failed.append(dataset)
    
    return len(failed) == 0, failed


def test_preprocessing():
    """Test preprocessing functionality."""
    print("\n🔍 Testing preprocessing...")
    
    try:
        from preprocess import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        test_text = "Check out this amazing product! http://example.com @user #hashtag"
        
        tokens = preprocessor.preprocess_text(test_text)
        
        if tokens and len(tokens) > 0:
            print(f"  ✅ Text preprocessing works")
            print(f"     Input: '{test_text}'")
            print(f"     Output: {tokens}")
            return True
        else:
            print(f"  ❌ Preprocessing returned empty tokens")
            return False
    
    except Exception as e:
        print(f"  ❌ Preprocessing failed: {e}")
        return False


def test_topic_modeling():
    """Test topic modeling functionality."""
    print("\n🔍 Testing topic modeling...")
    
    try:
        from model_utils import TopicModeler
        
        # Create sample documents
        documents = [
            ['machine', 'learning', 'artificial', 'intelligence'],
            ['data', 'science', 'analytics', 'statistics'],
            ['python', 'programming', 'coding', 'development']
        ]
        
        modeler = TopicModeler(num_topics=2, passes=5)
        modeler.train_lda(documents)
        
        topics = modeler.get_topic_keywords(num_words=3)
        
        if topics and len(topics) > 0:
            print(f"  ✅ Topic modeling works")
            print(f"     Generated {len(topics)} topics")
            return True
        else:
            print(f"  ❌ Topic modeling failed to generate topics")
            return False
    
    except Exception as e:
        print(f"  ❌ Topic modeling failed: {e}")
        return False


def test_sample_data():
    """Test if sample data exists and is valid."""
    print("\n🔍 Testing sample data...")
    
    try:
        import pandas as pd
        import os
        
        if not os.path.exists('sample_data.csv'):
            print(f"  ❌ sample_data.csv not found")
            return False
        
        df = pd.read_csv('sample_data.csv')
        
        required_cols = ['username', 'text', 'followers_count', 'friends_count', 'statuses_count', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False
        
        print(f"  ✅ Sample data is valid")
        print(f"     Rows: {len(df)}")
        print(f"     Genuine: {(df['label'] == 0).sum()}")
        print(f"     Fake: {(df['label'] == 1).sum()}")
        return True
    
    except Exception as e:
        print(f"  ❌ Sample data test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Twitter Fake Account Detection - Module Tests")
    print("=" * 60)
    
    results = []
    
    # Test imports
    success, failed = test_imports()
    results.append(('Required Modules', success, failed))
    
    # Test custom modules
    success, failed = test_custom_modules()
    results.append(('Custom Modules', success, failed))
    
    # Test NLTK data
    success, failed = test_nltk_data()
    results.append(('NLTK Data', success, failed))
    
    # Test preprocessing
    success = test_preprocessing()
    results.append(('Preprocessing', success, []))
    
    # Test topic modeling
    success = test_topic_modeling()
    results.append(('Topic Modeling', success, []))
    
    # Test sample data
    success = test_sample_data()
    results.append(('Sample Data', success, []))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, success, failed in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if failed:
            print(f"  Failed items: {', '.join(failed)}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed! You're ready to run the application.")
        print("\n🚀 Start the app with: streamlit run app.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the app.")
        print("\n💡 Try running: python setup.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
