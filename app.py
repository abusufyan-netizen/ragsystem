import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- Page Configuration ---
st.set_page_config(page_title="Free Medical RAG Q&A Generator", page_icon="🩺", layout="wide")
st.title("🩺 Medical Textbook Q&A Generator (100% Free & Local)")
st.caption("No API keys required • Powered by HuggingFace & Ollama")

# --- Sidebar Setup ---
with st.sidebar:
    st.header("⚙️ Local Settings")
    
    # Model Selection
    model_name = st.selectbox(
        "Select Ollama Model",
        ["llama3.2", "meditron", "mistral", "phi3"],
        help="Make sure you have pulled this model in Ollama (e.g., `ollama pull llama3.2`)"
    )
    
    st.divider()
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])

# Cache embeddings builder (uses CPU, no API key needed)
@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Split medical context into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    # Open-source CPU embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.remove(tmp_path)
    return vectorstore

# --- Main App Interface ---
if uploaded_file:
    with st.spinner("Processing medical PDF with free embeddings model..."):
        try:
            vectorstore = build_vector_store(uploaded_file.getvalue())
            st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.stop()

    st.divider()
    st.subheader("🎯 Question Generation Requirements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        qa_type = st.selectbox(
            "Format",
            ["USMLE Multiple Choice (MCQ)", "Anki Flashcards (Front/Back)", "Short Answer Vignettes", "True / False with Rationale"]
        )
    
    with col2:
        difficulty = st.selectbox(
            "Target Audience",
            ["Medical Student", "USMLE Step 1 / Step 2", "Clinical Resident", "General Study"]
        )
        
    with col3:
        num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=3)
        
    topic_focus = st.text_input(
        "Topic / Focus Keyword (Optional)", 
        placeholder="e.g., Cardiology, Pharmacology, Pathology, Treatment guidelines"
    )

    if st.button("🚀 Generate Questions", type="primary"):
        with st.spinner("Querying vector store and generating questions via local LLM..."):
            
            # Search relevant chunks
            search_query = topic_focus if topic_focus else "medical concepts guidelines diagnoses"
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(search_query)
            
            context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Local Medical Educator Prompt
            prompt_template = """You are a medical exam creator. 
Using ONLY the medical text context below, generate high-quality assessment questions matching the specific requirements.

Medical Context:
{context}

Requirements:
- Format: {qa_type}
- Difficulty Level: {difficulty}
- Number of Questions: {num_questions}
- Focus Topic: {topic_focus}

Instructions:
1. Ensure full clinical precision based only on the provided context.
2. For MCQs, provide 4 options (A-D), mark the correct answer, and give rationales.
3. For Flashcards, use distinct 'Front:' and 'Back:' labels.
"""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Initialize Local LLM via Ollama
            try:
                llm = ChatOllama(model=model_name, temperature=0.2)
                chain = prompt | llm
                
                response = chain.invoke({
                    "context": context_text,
                    "qa_type": qa_type,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "topic_focus": topic_focus if topic_focus else "General textbook overview"
                })
                
                # Display Output
                st.markdown("### 📋 Generated Assessment")
                st.markdown(response.content)
                
                # Download Button
                st.download_button(
                    label="📥 Download Q&A (.txt)",
                    data=response.content,
                    file_name="medical_generated_qa.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error connecting to local LLM: {e}")
                st.info("💡 Tip: Ensure Ollama is running on your computer (`ollama serve`) and the model is downloaded (`ollama pull llama3.2`).")

else:
    st.info("👈 Please upload a medical textbook PDF in the sidebar to start.")
          
