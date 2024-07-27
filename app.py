import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber
import io
from PIL import Image

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")  # Replace with your API key environment variable
if api_key is None:
    st.error("API Key not found. Please set it in the .env file.")
    st.stop()

genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    """Extract text from the uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def display_pdf(pdf_docs):
    """Display the pages of the PDF documents as images."""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    img = page.to_image(resolution=200)  # Adjust resolution as needed
                    img = img.original
                    # Convert image to BytesIO object
                    image_bytes = io.BytesIO()
                    img.save(image_bytes, format='PNG')
                    image_bytes.seek(0)
                    st.image(image_bytes, use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying {pdf.name}: {e}")

def get_text_chunks(text):
    """Split the text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create a vector store from text chunks and save it locally."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain():
    """Create a conversational chain for answering questions based on context."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say, "answer is not available in the context." Provide all relevant details.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")

def user_input(user_question):
    """Process the user's question and retrieve an answer from the PDF context."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing question: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", page_icon=":books:")
    st.header("DocQuest - Chat with PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        # Process uploaded PDFs
        if st.button("Submit & Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.pdf_docs = uploaded_files  # Store the PDF names in session state
                        st.success("Done")
                    else:
                        st.error("No text extracted from PDFs.")
        
        # Display uploaded PDFs
        if st.session_state.pdf_docs:
            st.markdown("### Currently Opened PDFs:")
            for pdf in st.session_state.pdf_docs:
                st.markdown(f"- *{os.path.basename(pdf.name)}*")
            display_pdf(st.session_state.pdf_docs)

    # Main area for questions and conversation
    user_question = st.chat_input("Ask a question about the PDFs")

    # Process user question if available
    if user_question and st.session_state.pdf_docs:
        answer = user_input(user_question)
        st.session_state.conversation.append({"question": user_question, "answer": answer})

    st.write("### Conversation History")
    for item in st.session_state.conversation:
        st.markdown(f"""
            <div style="padding: 10px; background-color: gray; border-radius: 5px;">
                <strong>Question:</strong> {item['question']}
            </div>
            <div style="padding: 10px; border-radius: 5px;">
                <strong>Answer:</strong> {item['answer']}
            </div>
            <br>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
