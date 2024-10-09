import os
import logging
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_openai import OpenAIEmbeddings  # Updated import from langchain-openai package
from langchain_openai.chat_models import ChatOpenAI  # Use ChatOpenAI for chat-based models like GPT-4
import store_documents  # Assuming this is the name of your storedocuments.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load environment variables from .env file
load_dotenv()

def create_qa_chain():
    try:
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OpenAI API key is missing. Please set it in your .env file.")
            return None

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        logging.info("Initialized OpenAI embeddings.")

    except Exception as e:
        logging.error(f"Error initializing OpenAI embeddings: {e}")
        return None

    # Define the directory where the vector store is persisted
    persist_directory = os.getenv("PERSIST_DIRECTORY")

    # Check if the vector store exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        logging.info("Vector store is not empty. Proceeding to load it.")
    else:
        logging.info("Vector store is empty or missing. Populating it with documents.")
        try:
            store_documents.store_documents()  # Call the function to store documents
            logging.info("Documents have been stored successfully.")
        except Exception as e:
            logging.error(f"Error in store_documents: {e}")
            return None

    # Initialize Chroma with the persisted directory and embeddings
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logging.info("Initialized Chroma vector store.")
    except Exception as e:
        logging.error(f"Error initializing Chroma vector store: {e}")
        return None

    # Set up the retriever using Chroma's vector store
    try:
        retriever = vectorstore.as_retriever()
        logging.info("Retriever has been set up successfully.")
    except Exception as e:
        logging.error(f"Error setting up retriever: {e}")
        return None

    # Initialize the OpenAI LLM with the desired model
    try:
        llm = ChatOpenAI(
            temperature=0.0,
            openai_api_key=api_key,
            model="gpt-4o-mini"  # Changed model from "gpt-4" to "4o-mini"
        )
        logging.info("Initialized ChatOpenAI with 'o1-mini' model.")
    except Exception as e:
        logging.error(f"Error initializing ChatOpenAI: {e}")
        return None

    # Create the RetrievalQA chain
    try:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        logging.info("RetrievalQA chain has been created successfully.")
    except Exception as e:
        logging.error(f"Error creating RetrievalQA chain: {e}")
        return None

    return qa_chain

if __name__ == "__main__":
    qa_chain = create_qa_chain()
    if qa_chain is None:
        logging.error("Failed to create QA chain. Exiting the program.")
    else:
        query = "What is a universal function approximator?"
        try:
            result = qa_chain.invoke({"query": query})
            print(f"Answer: {result['result']}")
        except Exception as e:
            logging.error(f"Error invoking QA chain: {e}")
