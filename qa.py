"""
This function gets an answer to a question from a knowledge base created from a PDF file using OpenAI embeddings.

The function takes a knowledge base and a question as input and returns an answer.
"""

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import streamlit as st

def get_answer(chunks, question):
    docs = chunks.similarity_search(question)
    
    llm = OpenAI(model_name="text-curie-001")

    chain = load_qa_chain(llm, chain_type="stuff")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)

    return response
