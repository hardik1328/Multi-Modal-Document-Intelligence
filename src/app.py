import streamlit as st
import os
import sys

# Add the current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.ingest import PDFProcessor
from src.vector_store import VectorStore
from src.rag import RAGSystem
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide", page_title="Multi-Modal RAG")

st.title("Multi-Modal Document Intelligence")
st.markdown("### RAG-Based QA System with Text, Tables, and Images")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    if not os.getenv("GROQ_API_KEY"):
        api_key = st.text_input("Enter Groq API Key", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("API Key set!")
    else:
        st.success("API Key found in environment.")
    
    st.divider()
    
    # PDF Processing
    st.subheader("Document Ingestion")
    # Option to use default test file
    use_test_file = st.checkbox("Use Test File (test/qatar_test_doc.pdf)", value=True)
    uploaded_file = st.file_uploader("Or Upload PDF", type=['pdf'])
    
    if st.button("Process Document"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please set Groq API Key first.")
        else:
            with st.spinner("Processing... (Extracting Text & Analyzing Images) - This may take a moment for image captioning..."):
                try:
                    target_path = ""
                    if uploaded_file:
                        # Save uploaded file temporarily
                        target_path = f"temp_{uploaded_file.name}"
                        with open(target_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    elif use_test_file:
                        target_path = "test/qatar_test_doc.pdf"
                        if not os.path.exists(target_path):
                            st.error(f"Test file not found at {target_path}")
                            st.stop()
                    else:
                        st.error("Please upload or select a file.")
                        st.stop()
                        
                    processor = PDFProcessor(target_path)
                    content = processor.extract_content()
                    st.info(f"Extracted {len(content)} chunks/images.")
                    
                    # Indexing
                    vs = VectorStore()
                    vs.add_documents(content)
                    st.success("Indexing Complete! You can now ask questions.")
                    
                    # Cleanup temp
                    if uploaded_file and os.path.exists(target_path):
                        os.remove(target_path)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag = RAGSystem()
                response, sources = rag.query(prompt)
                st.markdown(response)
                
                # Show sources in expander
                with st.expander("View Retrieved Context"):
                    st.write(sources)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                 st.error(f"Error generating response: {e}")

