from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Initialize LLM
llm = ChatOllama(model="llama3.1:latest", temperature=0.8, streaming=True)

# Prompt template - blueprint for prompts, MessagesPlaceholder is a placeholder for the chat history
# PromptTemplate = builds the next request to the model
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and help with tasks. If the user wants to end the conversation, ask them to type 'quit' to end the chat"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{prompt}")
])

# Add chat history
chat_history = []

print("Welcome to the LLM chatbot! Type 'quit' to end the chat.")
while True:
    prompt = input("User: ").strip()
    if prompt.lower() == "quit" or prompt.lower() == "exit":
        break

    # prompt_template.invoke() returns list of messages including chat history and user/system prompts
    # Constructs the current request to the model
    formatted_messages = prompt_template.invoke({
        "chat_history": chat_history,
        "prompt": prompt
    })

    print("Response: ", end="", flush=True)

    # Stream response from LLM using chat_history
    for chunk in llm.stream(formatted_messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n" + "-" * 100 + "\n")

    # Add prompt & response to chat history
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=chunk.content))

print("Thank you for chatting! Goodbye!")