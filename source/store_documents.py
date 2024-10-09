import os
import pdfplumber
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import langchain_openai
from lecture_notes import download_lecture_notes  # Import the function
import logging


# Load environment variables
load_dotenv()

def store_documents():

    pdf_directory = os.getenv("PDF_DIRECTORY")
    persist_directory = os.getenv("PERSIST_DIRECTORY")


    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Check if the PDF directory exists
    if not os.path.exists(pdf_directory):
        logging.error(f"Directory '{pdf_directory}' does not exist. Please check the path.")
        return

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    # Download lecture notes if no PDFs are found
    if not pdf_files:
        logging.info("No PDF files found. Downloading lecture notes...")
        try:
            download_lecture_notes(pdf_directory)
            pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
            if not pdf_files:
                logging.error("No PDF files were found after downloading. Please check the download process.")
                return
        except Exception as e:
            logging.error(f"Failed to download lecture notes: {e}")
            return

    all_documents = []

    # Loop through all the PDFs in the directory
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Only process non-empty text
                        logging.info(f"Processing text from {pdf_file}, page {page.page_number}")
                        all_documents.append(Document(page_content=text))
                    else:
                        logging.warning(f"Empty text extracted from {pdf_file}, page {page.page_number}")
        except Exception as e:
            logging.error(f"Error processing '{pdf_file}': {e}")
            continue  # Skip to the next file

    if not all_documents:
        logging.error("No valid documents were processed from the PDFs. Please check the PDF content.")
        return

    # Create embeddings and store them in a vector database
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OpenAI API key is missing. Please set it in your .env file.")
            return

        embeddings = langchain_openai.OpenAIEmbeddings(openai_api_key=api_key)

        # Initialize Chroma with the persist_directory argument to auto-save the vectorstore
        vectorstore = Chroma.from_documents(
            all_documents,
            embeddings,
            persist_directory=persist_directory
        )
        logging.info("All PDFs have been successfully ingested and stored in the vector database.")
    except Exception as e:
        logging.error(f"Error creating embeddings or storing them in the vectorstore: {e}")

if __name__ == "__main__":
    store_documents()
