import streamlit as st

from extractTextAndBuildCorpus import build_corpus

from sentenceEmbeddingsFunctions import make_embeddings, calculate_cosine_embeddings
from tf_idf_Functions import preprocessCorpus , vectorize , calculate_cosine_similarity

from df_ai import ask_together_ai, makeDataFrame , generatePrompt






# Set page title
st.set_page_config(page_title="Assignment", layout="centered")
st.title("Candidate Recommendation Engine")
st.markdown("---")




# Job Description input
jd_text = st.text_area("Enter Job Description", height=200, placeholder="Paste the job description here...")

# Resume uploads
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)
st.markdown("---")





# Let user choose how many top resumes to display
top_n = st.slider("Select number of top resumes to display", min_value=1, max_value=10, value=5)

# Buttons for processing
st.markdown("### Choose Matching Method")
embedding_button = st.button("Match using Transformer based Sentence Embeddings", use_container_width=True)
tfidf_button = st.button("Match using TF-IDF Vectorization", use_container_width=True)

        
        
        




# Show Results Table and AI Summary
if embedding_button:
    if not jd_text.strip():
        st.toast("Please enter a job description.", icon="⚠️")
    elif not uploaded_files:
        st.toast("Please upload at least one resume file.", icon="📄")
    else:
        corpus = []
        scores = []
        
        # Compute Cosine Similarity Scores
        with st.spinner("Calculating Cosine Similary Using Sentence Embeddings"):
            corpus = build_corpus(jd_text, uploaded_files)
            embeddings = make_embeddings(corpus)
            scores = calculate_cosine_embeddings(embeddings)


        # Sort scores in descending order
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        resume_texts = corpus[1:]
        df = makeDataFrame(sorted_indices, resume_texts, uploaded_files, top_n)


        # Display DataFrame
        st.subheader(f"🏆 Top {top_n} Resume Matches (Sentence Embeddings)")
        st.dataframe(df, use_container_width=True, hide_index=True)


        # Display AI Summary
        prompt = generatePrompt(sorted_indices, resume_texts, jd_text)

        with st.spinner("Analyzing top resume with AI..."):
            summary = ask_together_ai(prompt)

        st.markdown("AI Evaluation of Top Resume")
        st.write(summary)







# Show Results Table and AI Summary
if tfidf_button:
    if not jd_text.strip():
        st.toast("Please enter a job description.", icon="⚠️")
    elif not uploaded_files:
        st.toast("Please upload at least one resume file.", icon="📄")
    else:
        corpus = []
        scores = []

        # Compute Cosine Similarity Scores
        with st.spinner("Calculating Cosine Similarity Using TF-IDF Vectors"):
            corpus = build_corpus(jd_text, uploaded_files)
            cleanedCorpus = preprocessCorpus(corpus)
            tfidf_matrix = vectorize(cleanedCorpus)
            scores = calculate_cosine_similarity(tfidf_matrix)      


        # Sort scores in descending order
        sorted_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        resume_texts = corpus[1:]
        df = makeDataFrame(sorted_indices, resume_texts, uploaded_files, top_n)


        # Display DataFrame
        st.subheader(f"🏆 Top {top_n} Resume Matches (TF-IDF)")
        st.dataframe(df, use_container_width=True, hide_index=True)


        # Display AI Summary
        prompt = generatePrompt(sorted_indices, resume_texts, jd_text)

        with st.spinner("Analyzing top resume with AI..."):
            summary = ask_together_ai(prompt)

        st.markdown("AI Evaluation of Top Resume")
        st.write(summary)