import os
import pdfplumber
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma  # Updated import for Chroma
import langchain_openai


def store_documents():
    # Directory containing all the downloaded PDFs
    pdf_directory = "C:/Users/sweth/Tutorify-AI/source/lecture notes"  # Adjust this path based on where your PDFs are located

    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"Directory '{pdf_directory}' does not exist. Please check the path.")

    # List to hold all the documents from multiple PDFs
    all_documents = []

    # Loop through all the PDFs in the directory
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_directory, pdf_file)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Check if the text was successfully extracted
                        all_documents.append(Document(page_content=text))

    # Create embeddings and store them in a vector database
    embeddings = langchain_openai.OpenAIEmbeddings(
        openai_api_key="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"
        # Replace with your OpenAI API key
    )

    # Initialize Chroma with the persist_directory argument to auto-save the vectorstore
    vectorstore = Chroma.from_documents(all_documents, embeddings,
                                        persist_directory="C:/Users/sweth/Tutorify-AI/source/vectorstore")

    print("All PDFs have been successfully ingested and stored in the vector database.")


if __name__ == "__main__":
    store_documents()
