from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity




def make_embeddings(corpus):

    # Generates sentence embeddings using a pre-trained transformer model.
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_tensor=False)
    return embeddings



def calculate_cosine_embeddings(embeddings):
    """
    # Calculates cosine similarity between JD embedding and resume embeddings. 
    # Returns a 1D list of similarity scores.
    """
    jd_embedding = embeddings[0].reshape(1, -1)
    resume_embeddings = embeddings[1:]
    cosine_scores = cosine_similarity(jd_embedding, resume_embeddings)[0]
    return cosine_scores