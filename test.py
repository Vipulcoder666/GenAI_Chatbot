import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# .env file se API key load karo
load_dotenv()

# Groq LLM ko initialize karo
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Simple test message bhejo
response = llm.invoke("Hello! Ek line mein bata do tum kaun ho")
print(response.content)