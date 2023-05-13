# Ask Your PDF
This is a Python application that allows users to ask questions about PDF documents and get answers using OpenAI.

### NB 
Current implementation is using text-davinci-003 ( next iteration will be using chat-gpt API , reducing cost.

## Getting Started
### Prerequisites
- Python 3.7 or later
- Streamlit
- PyPDF2
- Langchain
### Installation
1. Clone the repository
   <pre><code>git clone https://github.com/SimonMagusPY/AskPDF.git</code></pre>
   
2. Install the required packages:
    <pre><code>pip install -r requirements.txt</code></pre>
    
### Usage
1. Run the app:
   <pre><code>streamlit run app.py</code></pre>
2. Upload a PDF file.
3. Ask a question about the PDF document.
4. Get the answer from the OpenAI model.

## Contributing
Contributions are welcome! If you have any suggestions or find any bugs, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
