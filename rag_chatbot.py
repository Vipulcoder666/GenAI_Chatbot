import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Step 1: PDF Load karo
loader = PyPDFLoader("test_document.pdf")
documents = loader.load()
print(f"✅ PDF load hui. Total pages: {len(documents)}")

# Step 2: Chunks mein todo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Document {len(chunks)} chunks mein tut gaya")

# Step 3: Embeddings model banao (free, local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Vector Store banao (chunks ke embeddings store honge yahan)
vector_store = FAISS.from_documents(chunks, embeddings)
print("✅ Vector Store ban gaya")

# Step 5: Retriever banao (jo similar chunks dhundega)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 6: LLM Initialize karo
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

print("\n🤖 RAG Chatbot Ready! Apne PDF se related sawaal poocho ('exit' likho band karne ke liye)\n")

while True:
    query = input("Tum: ")
    
    if query.lower() == "exit":
        print("Bot: Bye-Bye! 👋")
        break
    
    # Step 7: Query se related chunks dhundo
    relevant_chunks = retriever.invoke(query)
    
    # Un chunks ka text nikal ke jodo
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Step 8: Prompt banao (context + user ka sawaal)
    prompt = f"""Answer the user's question based on the context given below.
Respond in the SAME language that the user used to ask the question 
(if they ask in English, reply in English; if they ask in Hindi/Hinglish, reply in Hindi/Hinglish).
If the answer is not found in the context, say so in the same language as the question.

Context:
{context}

Question: {query}

Answer:"""
    
    # Step 9: LLM se jawaab lo
    response = llm.invoke(prompt)
    print("Bot:", response.content)
    print()