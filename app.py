import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    # Load environment variables from a .env file
    load_dotenv()

    # Streamlit configuration
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # File uploader for PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Extract text from PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User input for question
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            # Use RetrievalQA or appropriate chain setup
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                try:
                    response = chain.run(input_documents=docs, question=user_question)
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
