import nltk
import subprocess
import sys

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def download_spacy_model():
    """Download required spaCy model"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully downloaded spaCy model")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")

if __name__ == "__main__":
    download_nltk_data()
    download_spacy_model() 