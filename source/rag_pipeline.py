from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_openai import ChatOpenAI  # Updated import for ChatOpenAI
import langchain_openai

def create_qa_chain():
    # Define the embedding function
    embeddings = langchain_openai.OpenAIEmbeddings(
        openai_api_key="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"  # Replace with your OpenAI API key
    )

    # Load the previously stored vector database with the embedding function
    vectorstore = Chroma(
        persist_directory="C:/Users/sweth/Tutorify-AI/source/vectorstore",  # Same path used in store_documents.py
        embedding_function=embeddings  # Embedding function required for querying
    )

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0.7, openai_api_key="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")  # Replace with your OpenAI API key

    # Create and return the RetrievalQA chain
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

if __name__ == "__main__":
    # Create qa_chain globally so it can be accessed
    qa_chain = create_qa_chain()

    # Example query
    #query1 = "what is the equation for probability of a new data point with features x having label i in multi class logistic regression?"
    query = "what is universal function approximator?"

    # Use invoke instead of run
    result = qa_chain.invoke({"query": query})

    print(f"Answer: {result['result']}")
