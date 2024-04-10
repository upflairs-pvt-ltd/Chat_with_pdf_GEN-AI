#fix the code it is reading pdfs everytime
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import GooglePalm
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import os
import time
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

st.header("Chat With Pdfs")
st.subheader('You can get your answer from the provided PDFs.')

# Function to read the text from PDF files
def extract_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf = PdfReader(pdf)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_embedding():
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    return embedding

def get_vector_db(chunks, embedding):
    vectorstore = FAISS.from_texts(chunks, embedding=embedding)
    return vectorstore

def get_memory():
    memory1 = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return memory1

def get_llm():
    api_key = os.getenv('GOOGLE_API_KEY')
    llm = GooglePalm(google_api_key=api_key, temperature=0.2)
    return llm

# Check if session state exists
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

pdf_docs = st.file_uploader(
    "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

if pdf_docs:
    # Process PDFs and store data in session state
    text = extract_text(pdf_docs)
    chunks = get_chunks(text)
    embedding = get_embedding()
    vector_store = get_vector_db(chunks, embedding)
    memory = get_memory()
    llm = get_llm()

    st.session_state.processed_data = {
        'chunks': chunks,
        'embedding': embedding,
        'vector_store': vector_store,
        'memory': memory,
        'llm': llm
    }

user_question = st.text_input('Ask your question:', key='user_question')
submit = st.button('Submit')

if submit:
    if st.session_state.processed_data:
        # Retrieve processed data from session state
        processed_data = st.session_state.processed_data

        # Retrieve components
        chunks = processed_data['chunks']
        embedding = processed_data['embedding']
        vector_store = processed_data['vector_store']
        memory = processed_data['memory']
        llm = processed_data['llm']

        # Process user question
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        response = chain.invoke({'question': user_question})
        st.write(response['answer'])
    else:
        st.write("Please upload PDFs and click 'Process' first.")
