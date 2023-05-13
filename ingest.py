"""
This function extracts chunks of text from a PDF file and creates a knowledge base of the text using OpenAI embeddings.

The function takes a PDF file as input and returns a knowledge base.
"""

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

"""
    Extracts chunks of text from a PDF file and creates a knowledge base of the text using OpenAI embeddings.

    Args:
        pdf: A PDF file.

    Returns:
        A knowledge base.
"""
def extract_chunks(pdf):
    # Check if the PDF file is valid
    if not pdf:
        raise ValueError('Please provide a valid PDF file.')
    
    # Extract the text from the PDF file
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    #Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #Create embeddings for the chunks
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base
