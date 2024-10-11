import os
import logging
import traceback
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import store_documents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load environment variables from .env file
load_dotenv()

def create_qa_chain():
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        logging.info("Initialized OpenAI embeddings.")

    except Exception as e:
        logging.error(f"Error initializing OpenAI embeddings: {e}")
        logging.error(traceback.format_exc())
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
            logging.error(traceback.format_exc())
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
        logging.error(traceback.format_exc())
        return None

    # Set up the retriever using Chroma's vector store
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        logging.info("Retriever has been set up successfully.")
    except Exception as e:
        logging.error(f"Error setting up retriever: {e}")
        logging.error(traceback.format_exc())
        return None

    # Initialize the OpenAI LLM with the desired model
    try:
        llm = ChatOpenAI(
            temperature=0.0,
            model_name="gpt-4o-mini"  # Updated model name as per your request
        )
        logging.info("Initialized ChatOpenAI with 'gpt-4o-mini' model.")
    except Exception as e:
        logging.error(f"Error initializing ChatOpenAI: {e}")
        logging.error(traceback.format_exc())
        return None

    # Set up conversation memory
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    except Exception as e:
        logging.error(f"Error initializing conversation memory: {e}")
        logging.error(traceback.format_exc())
        return None

    # Custom prompts to enhance conversation history utilization
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, incorporating any relevant information from the conversation history.

    Chat History:
    {chat_history}

    Follow-up question: {question}

    Rephrased question:
    """)

    QA_PROMPT = PromptTemplate.from_template("""
    You are an AI assistant that provides helpful answers to the user's questions.

    When generating your answer, you should consider both the provided context and the conversation history.

    If the answer is not contained within the context or the conversation history, politely inform the user that you don't know.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Helpful Answer:
    """)

    # Create the Conversational Retrieval Chain using from_llm with custom prompts
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT},
            verbose=True
        )
        logging.info("Conversational Retrieval Chain has been created successfully.")
    except Exception as e:
        logging.error(f"Error creating Conversational Retrieval Chain: {e}")
        logging.error(traceback.format_exc())
        return None

    return qa_chain

if __name__ == "__main__":
    try:
        qa_chain = create_qa_chain()
        if qa_chain is None:
            logging.error("Failed to create QA chain. Exiting the program.")
            exit(1)
        else:
            while True:
                query = input("You: ")
                if query.lower() in ["exit", "quit"]:
                    break
                try:
                    result = qa_chain({"question": query})
                    answer = result["answer"]
                    print(f"Assistant: {answer}")
                except Exception as e:
                    logging.error(f"Error invoking QA chain: {e}")
                    logging.error(traceback.format_exc())
                    print("Assistant: An error occurred while processing your query. Please try again.")
    except KeyboardInterrupt:
        print("\nExiting the program.")
        exit(0)
