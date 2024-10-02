import os
import pdfplumber
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import langchain_openai
from lecture_notes import download_lecture_notes  # Import the function

load_dotenv()

def store_documents():
    pdf_directory = "C:/Users/sweth/PycharmProjects/Tutorify-AI(git)/source/lecture notes"  # Adjust this path based on where your PDFs are located
    persist_directory = "C:/Users/sweth/PycharmProjects/Tutorify-AI(git)/course_content"

    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"Directory '{pdf_directory}' does not exist. Please check the path.")

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found. Downloading lecture notes...")
        download_lecture_notes(pdf_directory)
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

        if not pdf_files:
            raise FileNotFoundError("No PDF files were found after downloading. Please check the download process.")

    all_documents = []

    # Loop through all the PDFs in the directory
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Only process non-empty text
                        print(f"Processing text from {pdf_file}, page {page.page_number}")
                        all_documents.append(Document(page_content=text))
                    else:
                        print(f"Warning: Empty text extracted from {pdf_file}, page {page.page_number}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    if not all_documents:
        raise ValueError("No valid documents were processed from the PDFs. Please check the PDF content.")

    # Create embeddings and store them in a vector database
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set it in your .env file.")

        embeddings = langchain_openai.OpenAIEmbeddings(openai_api_key=api_key)

        # Initialize Chroma with the persist_directory argument to auto-save the vectorstore
        vectorstore = Chroma.from_documents(all_documents, embeddings, persist_directory=persist_directory)
        print("All PDFs have been successfully ingested and stored in the vector database.")
    except Exception as e:
        print(f"Error creating embeddings or storing them in the vectorstore: {e}")

if __name__ == "__main__":
    store_documents()
