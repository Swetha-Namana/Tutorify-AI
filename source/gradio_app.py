import gradio as gr
from rag_pipeline import qa_chain


def answer_question(query):
    return qa_chain.run(query)


interface = gr.Interface(fn=answer_question, inputs="text", outputs="text", title="AI Chatbot")
interface.launch()
