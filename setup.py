"""
Setup script for Twitter Fake Account Detection project
Handles installation of dependencies and NLTK data
"""

import subprocess
import sys
import nltk


def install_requirements():
    """Install required Python packages."""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data."""
    print("\n📚 Downloading NLTK data...")
    
    nltk_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    
    for package in nltk_packages:
        try:
            print(f"  Downloading {package}...")
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"  ⚠️ Warning: Could not download {package}: {e}")
    
    print("✅ NLTK data downloaded successfully!")


def create_directories():
    """Create necessary directories."""
    import os
    
    print("\n📁 Creating directories...")
    
    directories = ['models']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Created {directory}/")
        else:
            print(f"  {directory}/ already exists")
    
    print("✅ Directories created successfully!")


def main():
    """Main setup function."""
    print("=" * 60)
    print("🐦 Twitter Fake Account Detection - Setup")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print("\n🚀 To run the application, use:")
    print("   streamlit run app.py")
    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main()
