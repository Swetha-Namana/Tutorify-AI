import os
import pdfplumber
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import langchain_openai
from lecture_notes import download_lecture_notes  # Import the function

load_dotenv()
def store_documents():
    # Directory containing all the downloaded PDFs
    pdf_directory = "C:/Users/sweth/Tutorify-AI/source/lecture notes"  # Adjust this path based on where your PDFs are located

    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"Directory '{pdf_directory}' does not exist. Please check the path.")

    # Check if there are any PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    # If no PDFs found, download lecture notes
    if not pdf_files:
        print("No PDF files found. Downloading lecture notes...")
        download_lecture_notes(pdf_directory)  # Download PDFs
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]  # Re-check for PDFs

        # If still no PDFs, raise an error
        if not pdf_files:
            raise FileNotFoundError("No PDF files were found after downloading. Please check the download process.")

    # List to hold all the documents from multiple PDFs
    all_documents = []

    # Loop through all the PDFs in the directory
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Check if the text was successfully extracted
                        all_documents.append(Document(page_content=text))
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    # Create embeddings and store them in a vector database
    try:
        embeddings = langchain_openai.OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize Chroma with the persist_directory argument to auto-save the vectorstore
        vectorstore = Chroma.from_documents(all_documents, embeddings,
                                            persist_directory="C:/Users/sweth/Tutorify-AI/source/vectorstore")

        print("All PDFs have been successfully ingested and stored in the vector database.")
    except Exception as e:
        print(f"Error creating embeddings or storing them in the vectorstore: {e}")

if __name__ == "__main__":
    store_documents()
