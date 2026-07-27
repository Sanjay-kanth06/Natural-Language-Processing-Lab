import sys
import subprocess

# Ensure nltk is installed at runtime
try:
    import nltk
except ImportError:
    print("NLTK not found. Installing nltk...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"]) 
    import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

def process_text(text: str) -> None:
    if not text:
        print("No text provided. Exiting.")
        return

    sentences = sent_tokenize(text)

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    print("\nSentence-wise Processing:")
    print("=" * 60)
    print("Sentence Tokens:")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}. \"{sentence}\"")
    print("=" * 60)

    for i, sentence in enumerate(sentences, 1):
        words = word_tokenize(sentence)

        stemmed = [stemmer.stem(word) for word in words]
        lemmatized = [lemmatizer.lemmatize(word) for word in words]

        print(f"\nSentence {i}: {sentence}")
        print("  Stemmed words     :", ", ".join(f'"{word}"' for word in stemmed))
        print("  Lemmatized words  :", ", ".join(f'"{word}"' for word in lemmatized))
        print("  Token count       :", len(words))
    print("=" * 60)


if __name__ == '__main__':
    sample_text = "This is a sample sentence. NLTK installation test."

    # Prefer reading from the terminal when possible, but fall back safely if
    # the input stream is not available.
    try:
        text = input(
            "Enter a paragraph (press Enter to use the sample text): "
        ).strip()
        if not text:
            text = sample_text
    except EOFError:
        text = sample_text
    except Exception:
        text = sample_text

    process_text(text)