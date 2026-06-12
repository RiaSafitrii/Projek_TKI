import re
import nltk

from nltk.corpus import stopwords

# Download stopwords sekali saja
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    text = str(text).lower()

    # Hapus angka dan simbol
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenisasi
    words = text.split()

    # Stopword removal
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


if __name__ == "__main__":

    sample = """
    Harry Potter is a fantasy novel
    about a wizard school.
    """

    print(preprocess_text(sample))