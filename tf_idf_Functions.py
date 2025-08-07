# These functions are used while calculating the cosine similarity using TF-IDF

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')



import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------

def preprocessCorpus(corpus):
    """
    Takes a list of strings (documents), and returns a list of cleaned, preprocessed strings.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    processed_corpus = []

    for doc in corpus:
        # 1. Lowercase
        doc = doc.lower()

        # 2. Remove special characters, numbers, symbols (keep only words)
        doc = re.sub(r'[^a-z\s]', ' ', doc)

        # 3. Tokenize
        tokens = word_tokenize(doc)

        # 4. Remove stopwords and short words, apply stemming
        cleaned_tokens = [
            stemmer.stem(word) for word in tokens
            if word not in stop_words and len(word) > 2
        ]

        # 5. Join tokens back into string
        cleaned_text = ' '.join(cleaned_tokens)
        processed_corpus.append(cleaned_text)

    return processed_corpus



def vectorize(corpus):
    """
    Takes a list of preprocessed strings and returns the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix



def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculates cosine similarity between the JD vector (first row) and all resume vectors (rest).
    Returns a 1D list of similarity scores.
    """
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    cosine_scores = cosine_similarity(jd_vector, resume_vectors)[0]
    return cosine_scores