# Tutorify-AI
AI tool for automating quiz/assignment creation, with a chatbot for student queries and Canvas LMS integration.

This project is a web-based teaching assistant chatbot designed to help students with questions about an AI course. It uses AI lecture notes from UC Berkeley's CS188 course, stored in a vector database, to generate meaningful answers based on student queries. The chatbot is built using the OpenAI API, LangChain for document ingestion and vector storage, and Chroma for managing the vector database. The web interface is developed with Gradio.

## Features

- **Automated Lecture Notes Ingestion**: Downloads lecture notes in PDF format from [UC Berkeley's CS188 course site](https://inst.eecs.berkeley.edu/~cs188/su24/).
- **Document Processing**: Extracts and processes text from lecture notes using PDF parsing.
- **Vector Storage**: Stores processed documents in a Chroma vector database using OpenAI embeddings.
- **Retrieval-Augmented Generation (RAG)**: Utilizes a GPT-4 model to answer student queries based on stored lecture content.
- **Web-based Interface**: Provides an interactive user interface through Gradio for students to ask questions and get responses.

## Technologies Used

- **Python 3.x**
- **OpenAI API** (for GPT-4 and embeddings)
- **LangChain** (for document ingestion and chain management)
- **Chroma** (for vector database storage)
- **Gradio** (for the user interface)
- **pdfplumber** (for PDF parsing)
- **BeautifulSoup** (for web scraping)

## Prerequisites

Before running the project, ensure you have the following dependencies installed:

```bash
pip install openai langchain langchain_chroma langchain_openai gradio requests beautifulsoup4 pdfplumber python-dotenv
```
Additionally, you'll need to set up an .env file in your project directory with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key
```
##Project Structure
```bash

├── lecture_notes.py            # Script to download lecture notes
├── gradio.py                   # Script to launch the Gradio interface
├── ragpipeline.py              # Script to set up the RAG pipeline using OpenAI and Chroma
├── storedocuments.py           # Script to store lecture notes in vector database
├── .env                        # Contains environment variables (e.g., OpenAI API key)
```

#3How to Run the Project
1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
```
2. Set up the environment:

  - Ensure your .env file is in place with your OpenAI API key.

3. Download Lecture Notes:

  - Run the lecture_notes.py script to download and save the lecture notes:

```bash
python lecture_notes.py
```
4. Store Lecture Notes in Vector Database:

  - Ingest and store the downloaded lecture notes in the Chroma vector database:

```bash
python storedocuments.py
```
5. Launch the Chatbot:

  - Start the Gradio web interface for the chatbot:

```bash
python gradio.py
```
-This will open a local web interface where you can interact with the chatbot.

##Testing the Chatbot
- ###Sample Query: Try asking the chatbot a question like:

```bash
What is a universal function approximator?
```
Comprehensive Testing: Test the chatbot with various questions related to the AI course lecture notes to evaluate its performance.

##Future Improvements
- Message History Storage: Implement a feature to store user conversations and maintain chat history across sessions.
- Streamlit Interface: Optionally implement a Streamlit interface for a different user experience.
- Additional Questions: Extend the question set to improve chatbot coverage of course content.
- Error Handling: Improve the error handling for document processing and vector storage

##Acknowledgments
- UC Berkeley for the CS188 course materials
- OpenAI for providing the GPT-4 model and embeddings API
- LangChain for simplifying document management and vector storage
