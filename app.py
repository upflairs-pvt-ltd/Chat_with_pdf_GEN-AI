import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
# Function to read the text from PDF files
def extract_text(pdf_doc):
    text = ''
    for pdf in pdf_doc:
        pdf = PdfReader(pdf)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_vector_db(chunks):
    vectorstore = FAISS.from_texts(chunks)
    return vectorstore

st.header("Chat With Pdfs")
st.subheader('you can get your answer from the provided pdfs.')
pdf_docs = st.file_uploader(
    "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

user_question = st.text_input('Ask your question:',key='user_question')
submit = st.button('Submit')
if submit:
    text = extract_text(pdf_doc=pdf_docs)
    st.write(text[10000:20000])