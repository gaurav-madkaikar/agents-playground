from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os

"""
Multi-step chains are chains that are composed of multiple steps
- A chain is a pipeline of runnables.
- A chain can be composed of one or more runnables.
- LCEL becomes powerful when you build multiple steps.

LCEL becomes powerful when you build multiple steps

Example Goal: Generate a explanation of a topic and summarize it into a bulleted list
User provides topic
   ↓
Step 1: generate explanation
   ↓
Step 2: summarize explanation in bullets
   ↓
Step 3: provide a list of questions to help the user understand the topic
"""

load_dotenv()
MODEL = os.getenv("OLLAMA_MODEL")
MODEL_TEMPERATURE = os.getenv("MODEL_TEMPERATURE")
MODEL_STREAMING = os.getenv("MODEL_STREAMING")
MODEL_REASONING = os.getenv("MODEL_REASONING").lower() == "true"

llm = ChatOllama(model=MODEL, temperature=MODEL_TEMPERATURE, streaming=MODEL_STREAMING, reasoning=MODEL_REASONING)
explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI tutor than explains concepts in an easy and detailed manner."),
    ("human", "Explain topic {topic} in detail.")
])

summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI tutor than summarises detailed explanations into concise points."),
    ("human", "Summarise the explanation provided as bulleted points: {explanation}.")
])

question_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI tutor than generates a list of questions to help the user understand the topic."),
    ("human", "Generate a list of questions to test the user's understanding of the provided summary: {summary}.")
])

explain_chain = explain_prompt | llm | StrOutputParser()
summary_chain = summarise_prompt | llm | StrOutputParser()
questions_chain = question_prompt | llm | StrOutputParser()

topic = input("Enter a topic: ")

# We can connect the chain together using RunnableLambdas or do it manually
# RunnableLambda is a function that takes a chain and returns a new chain
# The new chain is a pipeline of the original chain and the RunnableLambda
explain_response = explain_chain.invoke({
    "topic": topic
})

summary_response = summary_chain.invoke({
    "explanation": "".join(explain_response)
})

questions_response = questions_chain.invoke({
    "summary": "".join(summary_response)
})

"""
# We can combine individual chain into a single chain using RunnableLambda to extract the individual chain outputs
combined_chain = (
    explain_chain |
    RunnableLambda(lambda explanation: {"explanation": explanation}) |
    summary_chain
)

summary = combined_chain.invoke({
    "topic": topic
})

print(summary) # Prints the summary string
"""


# A stream iterator can only be consumed once, so we need to print the chunks after the whole pipeline has been processed, or save the chunks to a separate list
def print_chunks(response):
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n\n" + "-" * 100 + "\n")

print_chunks(explain_response)
print_chunks(summary_response)
print_chunks(questions_response)