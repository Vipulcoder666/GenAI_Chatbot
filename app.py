import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Streamlit Cloud Secrets ko environment variable mein daalo
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Vipul's Chatbot", page_icon="🤖")
st.title("🤖 Upload your PDF..")

# Session state mein chat history store karo (Streamlit reload hone par bhi yaad rahe)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar mein PDF upload karne ka option
with st.sidebar:
    st.header("📄 Upload your PDF..")
    uploaded_file = st.file_uploader("choose your PDF", type="pdf")
    
    if uploaded_file is not None and st.session_state.vector_store is None:
        with st.spinner("PDF is processing....please wait"):
            # Uploaded file ko temporarily save karo
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # PDF load karo
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            
            # Chunks mein todo
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            # Embeddings aur Vector Store banao
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            
        st.success("✅ PDF is Ready! You can ask your questions..")

# LLM Initialize karo
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# Purani chat history dikhao
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input lo (chat box)
user_query = st.chat_input("Right your question here...")

if user_query:
    # User ka message history mein add karo aur dikhao
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    
    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            st.write("⚠️ First upload your PDF from sidebar!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon..."):
                # Retriever se relevant chunks dhundo
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_chunks = retriever.invoke(user_query)
                context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
                
                # Prompt banao
                prompt = f"""Answer the user's question based on the context given below.
Respond in the SAME language that the user used to ask the question.
If the answer is not found in the context, say so in the same language as the question.

Context:
{context}

Question: {user_query}

Answer:"""
                
                response = llm.invoke(prompt)
                st.write(response.content)
        
        # Bot ka response bhi history mein add karo
        st.session_state.messages.append({"role": "assistant", "content": response.content})