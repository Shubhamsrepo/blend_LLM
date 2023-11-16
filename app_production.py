import streamlit as st 
from streamlit_chat import message
import tempfile
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

def load_data(path):
    df = pd.read_csv(path)  
    return df

def calculate_chunk_size(df):
    column_names = df.columns
    first_row_values = df.iloc[0]
    total_chars_in_column_names = sum(len(name) for name in column_names)
    total_chars_in_first_row = sum(len(str(value)) for value in first_row_values)
    chunk_size = int((total_chars_in_column_names + total_chars_in_first_row)*1.4)
    return chunk_size

def load_and_split_data(path, chunk_size):
    loader = CSVLoader(file_path=path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

def create_embeddings(text_chunks, model_name):
    embeddings = HuggingFaceEmbeddings(model_name = model_name)
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    return docsearch

def save_embeddings(docsearch, DB_FAISS_PATH):
    docsearch.save_local(DB_FAISS_PATH)

def load_llm_model(model_path, config):
    llm = CTransformers(model=model_path,
                        model_type="llama",
                        config=config)
    return llm

def create_conversational_chain(llm, retriever):
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    return qa

def conversational_chat(qa, query, history):
    result = qa({"question": query, "chat_history":history})
    history.append((query, result["answer"]))
    return result["answer"], history

def main():
    st.title("Dataset Parser")
    uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        path = tmp_file_path
        model_path = 'llama-2-7b-chat.ggmlv3.q8_0.bin'
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        config = {"max_new_tokens": 2048, "context_length": 4096 , "temperature": 0}

        df = load_data(path)
        chunk_size = calculate_chunk_size(df)
        text_chunks = load_and_split_data(path, chunk_size)
        docsearch = create_embeddings(text_chunks, model_name)
        save_embeddings(docsearch, DB_FAISS_PATH)
        llm = load_llm_model(model_path, config)
        qa = create_conversational_chain(llm, docsearch.as_retriever())

        history = []
        past = ["Hey !"]
        generated = ["Hello ! Ask me anything about " + uploaded_file.name]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
            
            if submit_button and user_input:
                output, history = conversational_chat(qa, user_input, history)
                past.append(user_input)
                generated.append(output)

        if generated:
            with response_container:
                for i in range(len(generated)):
                    message(past[i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(generated[i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
