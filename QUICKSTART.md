# 🚀 Quick Start Guide

Get up and running with the Twitter Fake Account Detection dashboard in 5 minutes!

## ⚡ Fast Setup

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
python setup.py

# Start the application
streamlit run app.py
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (run in Python)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Create models directory
mkdir models

# Start the application
streamlit run app.py
```

## 📊 Using the Sample Dataset

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload the sample dataset**
   - Click "Browse files" in the sidebar
   - Select `sample_data.csv` from the project directory
   - Wait for preprocessing (should take 10-30 seconds)

3. **Train the models**
   - Keep default settings (6 topics, 0.2 test size)
   - Click "🚀 Start Training"
   - Wait for training to complete (1-2 minutes)

4. **Explore the results**
   - View topic word clouds
   - Check classification metrics
   - Examine feature importance

5. **Try predictions**
   - Scroll to "Try Prediction" section
   - Enter a tweet like: "Check out this amazing deal! http://spam.com"
   - Set followers: 100, friends: 5000, statuses: 20000
   - Click "🔍 Predict"

## 🎯 Expected Results with Sample Data

With the provided `sample_data.csv` (120 users, 60 genuine + 60 fake):

- **Accuracy**: ~85-95%
- **Precision**: ~85-95%
- **Recall**: ~85-95%
- **F1-Score**: ~85-95%
- **ROC-AUC**: ~90-98%

## 💡 Tips

### For Best Performance:
- Use at least 100 users in your dataset
- Ensure balanced classes (similar number of fake and genuine accounts)
- Include diverse tweet content
- Provide accurate behavioral features

### Common Patterns Detected:
- **Fake accounts** often have:
  - High friends-to-followers ratio
  - Excessive use of URLs and mentions
  - Promotional/spam language
  - Very high tweet counts

- **Genuine accounts** often have:
  - Balanced follower/friend ratios
  - Natural conversational language
  - Moderate activity levels
  - Diverse topic discussions

## 🔧 Troubleshooting

### Application won't start?
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Models training slowly?
- Reduce number of topics (try 4 instead of 6)
- Use a smaller dataset for testing
- Close other applications to free up RAM

### Poor accuracy?
- Check if dataset is balanced
- Increase number of training samples
- Verify data quality (no empty texts)
- Adjust number of topics

## 📚 Next Steps

1. **Try your own data**
   - Format your CSV according to the specification
   - Upload and train on real Twitter data

2. **Experiment with parameters**
   - Try different numbers of topics (4-10)
   - Adjust test set size
   - Modify preprocessing in `preprocess.py`

3. **Save your models**
   - Click "Save Models" after training
   - Reload later with "Load Models"

4. **Customize the dashboard**
   - Edit `app.py` to add new features
   - Modify visualizations
   - Add custom metrics

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review code comments in source files
- Open an issue on GitHub

---

**Happy detecting! 🐦🔍**
