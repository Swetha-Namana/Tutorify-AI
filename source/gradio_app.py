import gradio as gr
from rag_pipeline import create_qa_chain  # Replace with your actual import

# Create a QA chain instance
qa_chain = create_qa_chain()


def answer_query(history, query):
    """Function to handle user queries and return answers."""
    if qa_chain:
        try:
            # Run the QA chain and get the result
            result = qa_chain.invoke({"query": query})
            history.append((query, result['result']))
            return history, history, ""
        except Exception as e:
            history.append((query, f"Error during query execution: {e}"))
            return history, history, ""
    else:
        return history, [("System", "QA chain is not initialized.")], ""


# Set up the Gradio interface with chatbot-like format
with gr.Blocks() as iface:
    gr.Markdown("<h1 align='center'>AI Course Teaching Assistant Chatbot</h1>")

    chatbot = gr.Chatbot(label="Chatbot")
    query_input = gr.Textbox(label="Textbox", placeholder="Ask a question...")

    with gr.Row():
        clear_button = gr.Button("Clear")


    def reset_chat():
        return [], []


    query_input.submit(answer_query, [chatbot, query_input], [chatbot, chatbot, query_input])
    clear_button.click(reset_chat, [], [chatbot, chatbot])

if __name__ == "__main__":
    iface.launch()
