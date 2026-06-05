"""
LangChain's current structured output docs describe this exact goal: instead of parsing natural language, you get structured data such as JSON objects or Pydantic models that your application can use directly.

With StrOutputParser we get string as the final output, but real applications demand structured data, often as json arrays or objects. For that we use structured outputs in langchain.

Three levels of structured output
1. Prompt-only JSON
You tell the model:
```
Respond in JSON.
```
This is easy, but unreliable.

Level 2: JSON parser
You use LangChain to parse the response as JSON.
If the model returns invalid JSON, parsing fails.

Why JsonOutputParser is useful but not enough?

- JsonOutputParser checks that output is valid JSON.
- But it does not strongly enforce your business rules.
- The expected structure could be:
```
{
    "sentiment": "positive"|"mixed"|"negative",
    "positive_points": list[str],
    "negative_points": list[str],
    "rating_out_of_5": int [0, 5]
}
```

Level 3: Pydantic parser
You define a Python schema.
LangChain parses the model output and validates field types and constraints.
This is the most useful for application development.

Some considerations while using Pydantic parser:
- Structured output is usually better with review_chain.invoke(...) rather than review_chain.stream({...})
    - Normal chatbot response → streaming
    - Structured extraction → invoke
- Be strict in system prompt to enforce the output format.
    - Return only valid JSON.
    - Do not include markdown.
    - Do not include explanations.

    - Please give JSON if possible.
- Workflow: prompt → model text → parser → validated object
- You can specify the format instructions in the prompt template using the format_instructions parameter.
    - Example:
    ```
    Format instructions: {format_instructions}
    ```
"""

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

import os

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL")
MODEL_TEMPERATURE = os.getenv("MODEL_TEMPERATURE")
MODEL_STREAMING = os.getenv("MODEL_STREAMING")
MODEL_REASONING = os.getenv("MODEL_REASONING").lower() == "true"

llm = ChatOllama(model=MODEL, temperature=MODEL_TEMPERATURE, streaming=MODEL_STREAMING, reasoning=MODEL_REASONING)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    You are a helpful assistant. 
    Please answer question in valid JSON format only. 
    Do not wrap the JSON in ```json blocks.
    Do not include any other text or markdown.
    """
    ),
    ("human", 
    """
    Analyze this product review.

    Review:
    {review}

    Output format instructions:
    {format_instructions}
    """
    )
])

class ReviewAnalysis(BaseModel):
    # fixed literal options
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the review",
    )
    # list of strings
    positive_points: list[str] = Field(
        description="List of positive points mentioned in the review",
    )
    # list of strings
    negative_points: list[str] = Field(
        description="List of negative points mentioned in the review",
    )
    # integer between 0 and 5
    rating_out_of_5: int = Field(
        description="Estimated rating out of 5 based on the review",
        ge=0,
        le=5
    )

chain = prompt | llm | StrOutputParser()
chain_json = prompt | llm | JsonOutputParser()

pydantic_parser = PydanticOutputParser(
    pydantic_object=ReviewAnalysis
)
chain_pydantic = prompt | llm | pydantic_parser
# Pydantic parser output schema
print("Format instructions: ", pydantic_parser.get_format_instructions())

review = input("Enter a product review: ")

response = chain.invoke({
    "review": review,
    "format_instructions": pydantic_parser.get_format_instructions()
})

response_json = chain_json.invoke({
    "review": review,
    "format_instructions": pydantic_parser.get_format_instructions()
})

response_pydantic = chain_pydantic.invoke({
    "review": review,
    "format_instructions": pydantic_parser.get_format_instructions()
})

print("\n" + "-" * 100 + "\n")

print("String Output:")
print(response)
print(type(response))

print("\n" + "-" * 100 + "\n")

print("JSON Output:")
print(response_json)
print(type(response_json))
print(f"Sentiment: {response_json['sentiment']} | Positive Points: {response_json['positive_points']} | Negative Points: {response_json['negative_points']} | Rating Out of 5: {response_json['rating_out_of_5']}")

print("\n" + "-" * 100 + "\n")

print("Pydantic Output:")
print(response_pydantic)
print(type(response_pydantic))
print(f"Sentiment: {response_pydantic.sentiment} | Positive Points: {response_pydantic.positive_points} | Negative Points: {response_pydantic.negative_points} | Rating Out of 5: {response_pydantic.rating_out_of_5}")

print("\n" + "-" * 100 + "\n")