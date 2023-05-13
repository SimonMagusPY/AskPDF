from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

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

            response = chain.run(input_documents=docs, question=user_question)            

            st.write(response)
        else:
            st.write("Please enter a question to get an answer.")        




if __name__ == '__main__':
    main()