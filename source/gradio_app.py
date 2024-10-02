import openai
import gradio as gr
from rag_pipeline import create_qa_chain  # Import your RAG pipeline from ragpipeline.py

# Initialize the QA chain for RAG
qa_chain = create_qa_chain()

# Initialize the conversation history with a system message (optional)
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]


# Function to handle the chat interaction using RAG pipeline
def chat_with_assistant(user_input, conversation_history):
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Use RAG pipeline instead of OpenAI API call
    response = qa_chain.invoke({"query": user_input})

    # Extract the assistant's response
    assistant_response = response['result']

    # Append the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Return the assistant's response and updated history
    return assistant_response, conversation_history


# Function to handle the Gradio interface interaction
def gradio_chat(user_input, history):
    # Rebuild conversation history from Gradio's history format (list of tuples)
    conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

    for user_msg, assistant_msg in history:
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": assistant_msg})

    # Get the assistant's response and updated conversation history
    assistant_response, conversation_history = chat_with_assistant(user_input, conversation_history)

    # Update Gradio's history format (list of tuples)
    history.append((user_input, assistant_response))

    return history, history


# Create Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()  # Display conversation
    message = gr.Textbox(placeholder="Type your message here...")  # Input box for user input
    clear = gr.Button("Clear")  # Button to clear chat


    # Define behavior for submitting user input
    def submit(user_input, history):
        return gradio_chat(user_input, history)


    # Define behavior for clearing chat
    def clear_chat():
        return [], []


    message.submit(submit, [message, chatbot], [chatbot, chatbot])  # When message is submitted
    clear.click(clear_chat, [], [chatbot])  # Clear button click behavior

# Launch the Gradio app
demo.launch()
