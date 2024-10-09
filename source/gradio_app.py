import gradio as gr
from rag_pipeline import create_qa_chain  # Replace with your actual import
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Create a QA chain instance
try:
    qa_chain = create_qa_chain()
    if qa_chain is None:
        logging.error("Failed to create QA chain. Exiting the application.")
        exit(1)
    else:
        logging.info("QA chain initialized successfully.")
except Exception as e:
    logging.error(f"Exception occurred while creating QA chain: {e}")
    exit(1)


def answer_query(history, query):
    """Function to handle user queries and return answers."""
    if qa_chain:
        try:
            # Run the QA chain and get the result
            result = qa_chain.invoke({"query": query})
            answer = result.get('result', 'No response.')
            history.append((query, answer))
            return history, history, ""
        except Exception as e:
            logging.error(f"Error during query execution: {e}")
            history.append((query, "An error occurred while processing your query. Please try again later."))
            return history, history, ""
    else:
        logging.error("QA chain is not initialized.")
        history.append((query, "The QA system is currently unavailable."))
        return history, history, ""


# Set up the Gradio interface with chatbot-like format
with gr.Blocks() as iface:
    gr.Markdown("<h1 align='center'>AI Course Teaching Assistant Chatbot</h1>")

    chatbot = gr.Chatbot(label="Chatbot")
    query_input = gr.Textbox(label="Textbox", placeholder="Ask a question...")

    with gr.Row():
        clear_button = gr.Button("Clear")

    def reset_chat():
        return [], []

    query_input.submit(
        answer_query,
        inputs=[chatbot, query_input],
        outputs=[chatbot, chatbot, query_input]
    )
    clear_button.click(reset_chat, [], [chatbot, chatbot])

if __name__ == "__main__":
    iface.launch()
