from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3.1:latest", temperature=0.8, streaming=True)
prompts = [
    "What is the capital of France?",
    "Explain the concept of quantum computing in brief and concise manner",
    "Redis vs Memcached: summarise the key differences",
    "What is the meaning of life?"
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    # response = llm.invoke(prompt) - streaming = False
    print("Response: ", end="", flush=True)
    for chunk in llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
    print("-" * 100)