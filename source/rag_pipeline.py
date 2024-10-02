from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_openai import OpenAIEmbeddings  # Updated import from langchain-openai package
from langchain_openai.chat_models import ChatOpenAI  # Use ChatOpenAI for chat-based models like GPT-4
import os
from dotenv import load_dotenv
import store_documents  # Assuming this is the name of your storedocuments.py

# Load environment variables from .env file
load_dotenv()

def create_qa_chain():
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Ensure you're using the right environment variable
    )

    # Define the directory where the vector store is persisted
    persist_directory = "C:/Users/sweth/Tutorify-AI(git)/course_content"

    # Check if the vector store exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Vector store is not empty. Proceeding to load it.")
    else:
        print("Vector store is empty. Calling store_documents to populate it.")
        store_documents.store_documents()  # Call the function to store documents

    # Initialize Chroma with the persisted directory and embeddings
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Set up the retriever using Chroma's vector store
    retriever = vectorstore.as_retriever()

    # Initialize the OpenAI LLM with GPT-4
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # Use API key from environment variables
        model="gpt-4"  # Specify GPT-4 here
    )

    # Create the RetrievalQA chain
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

if __name__ == "__main__":
    qa_chain = create_qa_chain()
    query = "What is a universal function approximator?"
    result = qa_chain.invoke({"query": query})
    print(f"Answer: {result['result']}")
