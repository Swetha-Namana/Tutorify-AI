from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
import langchain_openai
import os
from dotenv import load_dotenv
import store_documents  # Assuming this is the name of your storedocuments.py

load_dotenv()
def create_qa_chain():
    embeddings = langchain_openai.OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Replace with your actual API key
    )

    # Define the directory where the vector store is persisted
    persist_directory = "C:/Users/dell/Downloads/Tutorify-AI/src/VectorStore"

    # Check if the vector store is empty by verifying the persist directory
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Vector store is not empty. Proceeding to load it.")
    else:
        print("Vector store is empty. Calling store_documents to populate it.")
        store_documents.store_documents()  # Call the function to store documents

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))  # Replace with your actual API key

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# Define qa_chain globally

qa_chain = create_qa_chain()

# Example query
query = "what is universal function approximator?"

# Use invoke instead of run
result = qa_chain.invoke({"query": query})

print(f"Answer: {result['result']}")
