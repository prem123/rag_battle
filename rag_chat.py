import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os
import glob

# 1. Load environment variables
load_dotenv()

# 2. Page Configuration
st.set_page_config(
    page_title="📚 English Textbook Q&A",
    page_icon="📚",
    layout="wide"
)

# 3. Custom CSS for fancy styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .header-title {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .header-subtitle {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Stats cards */
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stats-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Question input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 15px 20px;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Answer box styling */
    .answer-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-top: 1rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .answer-text {
        font-size: 1.15rem;
        line-height: 1.8;
        color: #333;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
</style>
""", unsafe_allow_html=True)

# 4. Initialize LLM and Embeddings
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model_name="gpt-4o-mini", 
    temperature=0.0
)
embeddings = OpenAIEmbeddings()

# 5. Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title" style="color: #ffffff !important;">📚 6th Standard English Textbook Q&A</h1>
    <p class="header-subtitle" style="color: #ffffff !important;">Ask any question about your NCERT English textbook and get instant answers!</p>
</div>
""", unsafe_allow_html=True)

# 6. Load PDFs
pdf_folder = os.path.join(os.path.dirname(__file__), "6th_NCERT_Textbooks")
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

raw_text = ""
for pdf_path in pdf_files:
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
    except Exception as e:
        st.error(f"Error reading {os.path.basename(pdf_path)}: {str(e)}")

# 7. Stats display
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{len(pdf_files)}</div>
        <div class="stats-label">📄 PDF Files Loaded</div>
    </div>
    """, unsafe_allow_html=True)

# 8. Process text
if not raw_text.strip():
    st.error("Could not extract text from PDFs. They may be scanned images.")
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)

    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{len(chunks)}</div>
            <div class="stats-label">🧩 Text Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    if chunks:
        # Create Vector Store
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Setup QA Chain with more chunks retrieved
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 8}),
            return_source_documents=True,
        )

        # Question Section
        st.markdown("---")
        st.markdown("### 🔍 Ask Your Question")
        
        query = st.text_input(
            "Enter your question",
            placeholder="Type your question here... (e.g., What is the moral of the story?)",
            label_visibility="collapsed"
        )

        if query:
            with st.spinner("🤔 Analyzing your question..."):
                result = qa.invoke({"query": query})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
            
            st.markdown("### 💡 Answer")
            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-text">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show retrieved sources for debugging
            with st.expander("📚 View Retrieved Sources"):
                if source_docs:
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                        st.markdown("---")
                else:
                    st.write("No source documents retrieved.")
            
            # Add helpful tips
            with st.expander("💡 Tips for better answers"):
                st.markdown("""
                - Be specific in your questions
                - Mention chapter or story names if known
                - Ask about characters, themes, or morals
                - You can ask about word meanings too!
                """)
