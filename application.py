#flask based app
from flask import Flask, render_template, request, send_from_directory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import os,time
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Define the directory where your PDF files are stored
base_path = os.getcwd()
PDF_DIRECTORY = os.path.join(base_path,'pdf_directory')

def extract_text(PDF_DIRECTORY):
    text = ""
    FILE_NAMES = os.listdir(PDF_DIRECTORY)
    for i in range(len(FILE_NAMES)):
        FILE_PATH = os.path.join(PDF_DIRECTORY,FILE_NAMES[i])
        print(FILE_PATH)
        loader = PyPDFLoader(FILE_PATH)
        pdf_doc = loader.load()
        for page in pdf_doc:
            text += page.page_content
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
    vector_db_name = 'faiss_index'
    vectorstore = FAISS.from_texts(chunks,embedding=embedding)
    vectorstore.save_local(vector_db_name)
    return vectorstore

def get_memory():
    memory1 = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return memory1

def get_llm():
    api_key = os.getenv('GOOGLE_API_KEY')
    llm = GooglePalm(google_api_key=api_key,temparature=0.2)
    return llm

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        for uploaded_file in uploaded_files:
            if uploaded_file.filename != "":
                file_path = os.path.join(PDF_DIRECTORY, uploaded_file.filename)
                uploaded_file.save(file_path)
        


        with ThreadPoolExecutor() as executor:
            text = executor.submit(extract_text,PDF_DIRECTORY)
            text = text.result()

            chunks = executor.submit(get_chunks,text)
            chunks = chunks.result()

            embedding = get_embedding()

            vector_db = executor.submit(get_vector_db,chunks,embedding).result()


        # # Reading text from the pdf files
        # text = extract_text(PDF_DIRECTORY=PDF_DIRECTORY)
        # #get chunks
        # chunks = get_chunks(text=text)
        # embedding = get_embedding()

        # # intialized the text into FAISS vectorDB 
        # vector_db = get_vector_db(chunks=chunks,embedding=embedding)

        return render_template('question_answering.html')
    return "Error: No file selected or invalid request method."


@app.route('/ques_ans',methods=['GET','POST'])
def ques_ans():
    if request.method =='POST':
        ques = str(request.form['ques'])
        instructor_embeddings = get_embedding()
        llm = get_llm()
        memory = get_memory()

        vector_store = FAISS.load_local('faiss_index', instructor_embeddings)
        chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
        )

        llm_response = chain.invoke({'question':ques})
        answer = str(llm_response['answer'])
        return render_template('question_answering.html',answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
