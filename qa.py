from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def get_answer(chunks, question):
    docs = chunks.similarity_search(question)

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)

    return response
