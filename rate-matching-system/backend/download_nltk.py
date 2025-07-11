import nltk

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Download required NLTK data
        nltk.download('punkt')
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    download_nltk_data() 