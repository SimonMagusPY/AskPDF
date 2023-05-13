"""
This is a Streamlit app that allows users to upload a PDF file and ask a question
about its content. The app uses the PyPDF2 library to extract text from the PDF file,
and then uses OpenAI embeddings to create a knowledge base of the text. Users can then
ask a question about the text, and the app uses a question-answering chain to find
the most relevant answer from the knowledge base.

To run the app, simply run this script in a Python environment with the required
dependencies installed, and then navigate to the local URL provided by Streamlit.
"""
import streamlit as st
from ingest import extract_chunks
from qa import get_answer


def main():
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask your PDF 💬")
    
    # Upload the file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    # Extract the text into chunks
    if pdf is not None:
        chunks = extract_chunks(pdf)
        
        # Show user input for question
        user_question = st.text_input("Ask a question about your PDF:")
        
        # Get answer using OpenAI
        if user_question:
            response = get_answer(chunks, user_question)
            st.write(response)
        else:
            st.write("Please enter a question to get an answer.")


if __name__ == '__main__':
    main()
