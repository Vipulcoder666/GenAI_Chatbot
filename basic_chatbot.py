import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
 
load_dotenv()

# initialize the LLM
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.7
)

print("🤖 Chatbot shuru ho gaya! ('exit' likho band karne ke liye)\n")

# Infinite loop - jab tak user 'exit' na bole

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Bot: Good bye! 👋")
        break

    # LLM ko message bhejo aur response lo
    response = llm.invoke(user_input)
    print("Bot:", response.content)
    print() # ek khali line, formatting ke liye
      
