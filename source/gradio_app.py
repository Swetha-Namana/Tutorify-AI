import gradio as gr
import logging
import traceback
from rag_pipeline import create_qa_chain

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
    logging.error(traceback.format_exc())
    exit(1)

def answer_query(user_input, history):
    """Function to handle user queries and return answers."""
    if qa_chain:
        if history is None:
            history = []
        try:
            # Run the QA chain and get the result
            result = qa_chain({"question": user_input})
            answer = result.get('answer', 'No response.')
            # Append the new interaction to the chat history
            history.append((user_input, answer))
            return history, "", history
        except Exception as e:
            logging.error(f"Error during query execution: {e}")
            logging.error(traceback.format_exc())
            history.append((user_input, "An error occurred while processing your query. Please try again later."))
            return history, "", history
    else:
        logging.error("QA chain is not initialized.")
        history.append((user_input, "The QA system is currently unavailable."))
        return history, "", history

def reset_chat():
    return [], "", []

with gr.Blocks() as iface:
    gr.Markdown("<h1 align='center'>AI Course Teaching Assistant Chatbot</h1>")

    chatbot = gr.Chatbot()
    state = gr.State([])  # Initialize the chat history

    with gr.Row():
        with gr.Column(scale=8):
            user_input = gr.Textbox(
                show_label=False,
                placeholder="Ask a question...",
                lines=1
            )
        with gr.Column(scale=2):
            submit_button = gr.Button("Send")

    with gr.Row():
        clear_button = gr.Button("Clear Chat")

    submit_button.click(
        answer_query,
        inputs=[user_input, state],
        outputs=[chatbot, user_input, state]
    )

    user_input.submit(
        answer_query,
        inputs=[user_input, state],
        outputs=[chatbot, user_input, state]
    )

    clear_button.click(reset_chat, outputs=[chatbot, user_input, state])

if __name__ == "__main__":
    try:
        share_info = iface.launch(share=True, debug=True)
        if share_info and 'share_url' in share_info:
            logging.info(f"Shareable URL: {share_info['share_url']}")
            print(f"Shareable URL: {share_info['share_url']}")
        else:
            logging.warning("Shareable URL was not generated.")
            print("Shareable URL was not generated.")
    except Exception as e:
        logging.error(f"Failed to launch Gradio interface: {e}")
        logging.error(traceback.format_exc())
        print(f"Failed to launch Gradio interface: {e}")
