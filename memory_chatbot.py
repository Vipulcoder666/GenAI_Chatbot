import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature = 0.7
)

# Yahan poori conversation history store hogi

chat_history = []

print("🤖 Chatbot shuru ho gaya (Memory ke saath)! ('exit' likho band karne ke liye)\n")

while True:
    user_input = input("you: ")

    if user_input.lower() == "exit":
        print("Good bye!👋")
        break

    # User ka message history mein add karo
    chat_history.append(HumanMessage(content=user_input))

    # Poori history LLM ko bhejo (na ki sirf naya message)
    response = llm.invoke(chat_history)

    # Bot ka jawaab bhi history mein add karo
    chat_history.append(AIMessage(content=response.content))

    print("Bot:", response.content)
    print()

    