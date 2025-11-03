# 🚀 Complete Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: ~500MB for dependencies
- **Internet**: Required for initial setup

### Check Python Installation
```bash
python --version
# Should show: Python 3.8.x or higher
```

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3`
- **Linux**: `sudo apt-get install python3 python3-pip`

---

## Installation Options

### 🎯 Option 1: Quick Start (Recommended for Beginners)

**Windows Users:**
```bash
# 1. Double-click run.bat
# OR run in Command Prompt:
run.bat
```

**Mac/Linux Users:**
```bash
# 1. Run setup
python setup.py

# 2. Start application
streamlit run app.py
```

### 🔧 Option 2: Manual Installation (Recommended for Developers)

#### Step 1: Create Virtual Environment (Optional but Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

#### Step 4: Create Models Directory
```bash
# Windows:
mkdir models

# Mac/Linux:
mkdir -p models
```

#### Step 5: Run Application
```bash
streamlit run app.py
```

### 🐳 Option 3: Docker (Coming Soon)

---

## Verification

### Test Installation
```bash
python test_modules.py
```

**Expected Output:**
```
🧪 Twitter Fake Account Detection - Module Tests
============================================================
🔍 Testing module imports...
  ✅ Streamlit
  ✅ Pandas
  ✅ NumPy
  ✅ Gensim
  ✅ Scikit-learn
  ✅ NLTK
  ✅ Matplotlib
  ✅ Seaborn
  ✅ WordCloud
  ✅ Joblib
  ✅ Plotly

🔍 Testing custom modules...
  ✅ Text Preprocessor
  ✅ Model Utilities

🔍 Testing NLTK data...
  ✅ punkt
  ✅ stopwords
  ✅ wordnet
  ✅ omw-1.4

🔍 Testing preprocessing...
  ✅ Text preprocessing works

🔍 Testing topic modeling...
  ✅ Topic modeling works

🔍 Testing sample data...
  ✅ Sample data is valid

============================================================
📊 Test Summary
============================================================
Required Modules: ✅ PASSED
Custom Modules: ✅ PASSED
NLTK Data: ✅ PASSED
Preprocessing: ✅ PASSED
Topic Modeling: ✅ PASSED
Sample Data: ✅ PASSED
============================================================

✅ All tests passed! You're ready to run the application.

🚀 Start the app with: streamlit run app.py
```

---

## Troubleshooting

### Problem 1: "Python is not recognized"
**Cause**: Python not in system PATH

**Solution**:
1. Reinstall Python with "Add to PATH" checked
2. Or manually add Python to PATH:
   - Windows: System Properties → Environment Variables → Path
   - Mac/Linux: Add to `.bashrc` or `.zshrc`

### Problem 2: "pip is not recognized"
**Cause**: pip not installed or not in PATH

**Solution**:
```bash
python -m ensurepip --upgrade
```

### Problem 3: "Module not found" errors
**Cause**: Dependencies not installed

**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Problem 4: NLTK data errors
**Cause**: NLTK data not downloaded

**Solution**:
```bash
python setup.py
# OR manually:
python -c "import nltk; nltk.download('all')"
```

### Problem 5: "Permission denied" errors
**Cause**: Insufficient permissions

**Solution**:
```bash
# Windows: Run Command Prompt as Administrator
# Mac/Linux: Use sudo
sudo pip install -r requirements.txt
```

### Problem 6: SSL Certificate errors
**Cause**: Corporate firewall or proxy

**Solution**:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Problem 7: Memory errors during training
**Cause**: Insufficient RAM

**Solution**:
- Use smaller dataset
- Reduce number of topics (try 3-4 instead of 6)
- Close other applications
- Increase virtual memory

### Problem 8: Streamlit won't start
**Cause**: Port already in use

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Problem 9: Slow performance
**Cause**: Large dataset or limited resources

**Solution**:
- Use sample_data.csv for testing
- Reduce number of topics
- Reduce number of trees in Random Forest (edit model_utils.py)

### Problem 10: Import errors with spaCy
**Cause**: spaCy language model not downloaded

**Solution**:
```bash
python -m spacy download en_core_web_sm
# OR use NLTK only (spaCy is optional)
```

---

## Platform-Specific Notes

### Windows
- Use Command Prompt or PowerShell
- Backslashes in paths: `C:\Projects\Twikbot`
- Run `run.bat` for easy startup

### macOS
- Use Terminal
- Forward slashes in paths: `/Users/username/Twikbot`
- May need to install Xcode Command Line Tools: `xcode-select --install`

### Linux
- Use Terminal
- Forward slashes in paths: `/home/username/Twikbot`
- May need to install build essentials: `sudo apt-get install build-essential`

---

## Upgrading

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Update NLTK Data
```bash
python -c "import nltk; nltk.download('all', force=True)"
```

### Clear Cache
```bash
# Windows:
rmdir /s /q __pycache__
del /s *.pyc

# Mac/Linux:
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## Uninstallation

### Remove Virtual Environment
```bash
# Windows:
rmdir /s /q venv

# Mac/Linux:
rm -rf venv
```

### Remove Dependencies (if not using virtual environment)
```bash
pip uninstall -r requirements.txt -y
```

### Remove Project
```bash
# Delete the entire Twikbot folder
```

---

## Advanced Configuration

### Custom Streamlit Configuration
Copy `streamlit_config.toml` to `.streamlit/config.toml`:

```bash
# Windows:
mkdir .streamlit
copy streamlit_config.toml .streamlit\config.toml

# Mac/Linux:
mkdir -p .streamlit
cp streamlit_config.toml .streamlit/config.toml
```

### Modify Model Parameters
Edit `model_utils.py`:
- Line 17: Change `num_topics` default
- Line 18: Change `passes` for LDA
- Line 144: Change `n_estimators` for Random Forest

### Modify Preprocessing
Edit `preprocess.py`:
- Add custom stopwords
- Modify tokenization
- Add custom features

---

## Performance Optimization

### For Large Datasets
1. **Increase RAM allocation**
2. **Use sampling**: Process subset of data
3. **Reduce topics**: Use 3-4 topics instead of 6
4. **Reduce trees**: Use 50 trees instead of 100
5. **Use multiprocessing**: Already enabled with `n_jobs=-1`

### For Faster Training
```python
# In model_utils.py, modify:
TopicModeler(num_topics=4, passes=5)  # Reduce passes
RandomForestClassifier(n_estimators=50)  # Reduce trees
```

---

## Getting Help

### Documentation
- **README.md**: Complete project documentation
- **QUICKSTART.md**: Quick start guide
- **PROJECT_OVERVIEW.md**: Technical details
- **This file**: Installation help

### Testing
```bash
python test_modules.py  # Run diagnostic tests
```

### Community Support
- Open an issue on GitHub
- Check existing issues for solutions
- Contribute improvements via pull requests

---

## Next Steps

After successful installation:

1. ✅ **Run tests**: `python test_modules.py`
2. ✅ **Start app**: `streamlit run app.py`
3. ✅ **Upload sample data**: Use `sample_data.csv`
4. ✅ **Train models**: Click "Start Training"
5. ✅ **Explore results**: View visualizations
6. ✅ **Make predictions**: Test individual tweets
7. ✅ **Save models**: Click "Save Models"

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded
- [ ] Test script passes (`python test_modules.py`)
- [ ] Application starts (`streamlit run app.py`)
- [ ] Sample data loads successfully
- [ ] Models train without errors
- [ ] Visualizations display correctly
- [ ] Predictions work

---

**If all items are checked, you're ready to go! 🎉**

For quick reference, see [QUICKSTART.md](QUICKSTART.md)
