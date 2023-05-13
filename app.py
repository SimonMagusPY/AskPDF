from dotenv import load_dotenv
import os
import streamlit as st

def main(): 
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask your PDF 💬")

    pdf = st.file_uploader("Upload you PDF", type='pdf')
    
    

    


if __name__ == '__main__':
    main()