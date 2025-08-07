import docx2txt
from PyPDF2 import PdfReader


def extract_text_from_file(uploaded_file):
    """
    Extracts text from a Streamlit UploadedFile object depending on its type.
    Supports .txt, .pdf, .docx
    """
    file_type = uploaded_file.type

    # Handle TXT
    if file_type == "text/plain":
        return uploaded_file.read().decode("utf-8")

    # Handle DOCX
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return docx2txt.process(uploaded_file)

    # Handle PDF
    elif file_type == "application/pdf":
        text = ""
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    else:
        return ""



def build_corpus(jd_text, uploaded_files):
    """
    Constructs the corpus: [job_description, resume_1, resume_2, ...]
    """
    corpus = [jd_text.strip()]

    for file in uploaded_files:
        content = extract_text_from_file(file)
        corpus.append(content.strip())

    return corpus