from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOllama(model="llama3.1:latest", temperature=0.8, streaming=True)

# Add chat history
chat_history = []

# Set system personality
chat_history.append(SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks. If the user wants to end the conversation, ask them to type 'quit' to end the chat"))

print("Welcome to the LLM chatbot! Type 'quit' to end the chat.")
while True:
    prompt = input("User: ").strip()
    if prompt.lower() == "quit" or prompt.lower() == "exit":
        break
    # Add user prompt to chat history
    chat_history.append(HumanMessage(content=prompt))

    print("Response: ", end="", flush=True)

    # Stream response from LLM using chat_history
    for chunk in llm.stream(chat_history):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    # Add AI response to chat history
    chat_history.append(AIMessage(content=chunk.content))

    print("\n" + "-" * 100 + "\n")

print("Thank you for chatting! Goodbye!")