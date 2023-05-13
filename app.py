"""
This is a Streamlit app that allows users to upload a PDF file and ask a question
about its content. The app uses the PyPDF2 library to extract text from the PDF file,
and then uses OpenAI embeddings to create a knowledge base of the text. Users can then
ask a question about the text, and the app uses a question-answering chain to find
the most relevant answer from the knowledge base.

To run the app, simply run this script in a Python environment with the required
dependencies installed, and then navigate to the local URL provided by Streamlit.
"""

from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main(): 
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask your PDF 💬")
    
    #upload the file
    pdf = st.file_uploader("Upload you PDF", type='pdf')  
    #extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        #create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        #initialize docs to None
        docs = None 
        #show user input for question
        user_question = st.text_input("Ask a question about your PDF:")
    
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
        
        if docs:

            llm = OpenAI()    
            chain = load_qa_chain(llm, chain_type="stuff")     

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)            

            st.write(response)
        else:
            st.write("Please enter a question to get an answer.")        




if __name__ == '__main__':
    main()