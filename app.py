# streamlit base app
import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import os,time
from dotenv import load_dotenv
load_dotenv()
vector_temp = 0
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
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

    
def get_embedding():
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    return embedding
def get_vector_db(chunks,embedding):
    vectorstore = FAISS.from_texts(chunks,embedding=embedding)
    
    return vectorstore

def get_memory():
    memory1 = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return memory1

def get_llm():
    api_key = os.getenv('GOOGLE_API_KEY')
    llm = GooglePalm(google_api_key=api_key,temparature=0.2)
    return llm


st.header("Chat With Pdfs")
st.subheader('you can get your answer from the provided pdfs.')
pdf_docs = st.file_uploader(
    "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

user_question = st.text_input('Ask your question:',key='user_question')
submit = st.button('Submit')


if submit:
    text = extract_text(pdf_doc=pdf_docs)
    chunks = get_chunks(text)
    embedding = get_embedding()
    llm = get_llm()

    # To get some delay
    pregress_bar = st.progress(0)
    for i in range(101):
        time.sleep(0.01)
        pregress_bar.progress(i)

    vector_store = get_vector_db(chunks,embedding)
    memory = get_memory()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    response = chain.invoke({'question':user_question})
    st.write(response['answer'])