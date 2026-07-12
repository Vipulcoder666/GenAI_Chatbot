import os
import speech_recognition as sr
import pygame
import edge_tts
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

pygame.mixer.init()

async def _generate_speech(text, filename="response.mp3"):
    voice = "en-GB-RyanNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

def speak(text):
    print(f"Jarvis: {text}")
    asyncio.run(_generate_speech(text))
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()   # File ko release karo taaki dubara likhi ja sake

recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("\n🎤 Sun raha hoon... bolo")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            print("Kuch bola hi nahi, dubara try karo")
            return None
    
    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        print(f"Tumne bola: {text}")
        return text
    except sr.UnknownValueError:
        print("Samajh nahi aaya, dubara bolo")
        return None
    except sr.RequestError:
        print("Internet check karo")
        return None

print("📄 PDF load ho rahi hai...")
loader = PyPDFLoader("test_document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

print("✅ Jarvis Ready!\n")
speak("Jarvis ready hai. Apna sawaal poocho")

while True:
    user_text = listen()
    
    if user_text is None:
        continue
    
    query = user_text.strip()
    
    if query.lower() in ["exit", "band ho jao", "stop"]:
        speak("Alvida!")
        break
    
    relevant_chunks = retriever.invoke(query)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    prompt = f"""Answer the user's question based on the context given below in 1-2 short sentences (this will be spoken out loud, so keep it concise).
If the answer is not found in the context, say so briefly.

Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke(prompt)
    speak(response.content)