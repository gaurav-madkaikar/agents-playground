from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import os

"""
Runnables in langchain are objects that accept inputs and return outputs.
It is something that can be executed with:
.invoke()
.stream()
.batch()

Examples:
ChatPromptTemplate, ChatOllama, OutputParser, Chain are all runnables

Thus runnables can be composed into pipelines ('chains') using LCEL (LangChain Expression Language), which are runnables themselves.
```
# similar to unix pipes
chain = prompt | llm 
chain.stream({
    "chat_history": chat_history,
    "prompt": prompt
})
```

This is equivalent to calling:
```
formatted_messages = prompt_template.invoke({
    "chat_history": chat_history,
    "prompt": prompt
})
llm.stream(formatted_messages)
```
"""

load_dotenv()
MODEL = os.getenv("OLLAMA_MODEL")
MODEL_TEMPERATURE = os.getenv("MODEL_TEMPERATURE")
MODEL_STREAMING = os.getenv("MODEL_STREAMING")
MODEL_REASONING = os.getenv("MODEL_REASONING").lower() == "true"

# Initialize LLM
# reasoning=False disables Qwen3 "thinking" mode for faster direct responses
llm = ChatOllama(
    model=MODEL,
    temperature=MODEL_TEMPERATURE,
    streaming=MODEL_STREAMING,
    reasoning=MODEL_REASONING
)

# Prompt template - blueprint for prompts, MessagesPlaceholder is a placeholder for the chat history
# PromptTemplate = builds the next request to the model
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with a personal name of 'MindBot' that can answer questions and help with tasks. If the user wants to end the conversation at any point, only then should you ask them to type 'quit' to end the chat"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{prompt}")
])

# Create a reusable pipeline, | is the LCEL pipe operator
# StrOutputParser() is used to parse the response from the model into a string
# Currently model responses return AIMessage objects, so we need to parse them into a string
chain = prompt_template | llm | StrOutputParser()

# Add chat history
chat_history = []

print("Welcome to MindBot 🤖\nYour personal AI assistant!\n\nType 'quit' or 'exit' to end the conversation\n")
while True:
    prompt = input("User $> ").strip()
    if prompt.lower() == "quit" or prompt.lower() == "exit":
        break
    
    print("Bot  $> ", end="", flush=True)

    # Invoke the chain with the chat history and prompt rather than calling model explicitly
    # Returns list of strings, each string is a chunk of the response from the model
    response = chain.stream({"chat_history": chat_history, "prompt": prompt})

    for chunk in response:
        print(chunk, end="", flush=True)

    print("\n" + "-" * 100 + "\n")

    # Add prompt & response to chat history
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content="".join(response)))

print("Thank you for chatting with MindBot 🤖, until next time!")