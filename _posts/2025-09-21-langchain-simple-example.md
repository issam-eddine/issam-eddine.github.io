---
layout: post
title: langchain simple example
date: 2025-09-21 11:45:00
description: a simple example of how to use langchain
tags: code machine-learning
categories: learning
featured: true
related_posts: false
---

Let us learn the basics of Langchain using this github repository: [langchain-simple-example](https://github.com/issam-eddine/langchain-simple-example).

This repository showcases the use of langchain to create an agent that can respond to queries. It can use tools for searching Wikipedia and saving the results to a file.

A simple example of how to use langchain to invoke an LLM:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini")
response = llm.invoke("What is the capital of the moon?")
print(response.content)
```

Now, let us check how to create an agent that can use tools to search Wikipedia and save the results to a file – let us dive into the `main.py` file:

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import wikipedia_tool, save_tool

load_dotenv()


# Define the structure of the research response
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Initialize the language model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Initialize the output parser
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help with research queries.
            Provide a final response in this exact JSON format:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# List of tools available to the agent
tools = [wikipedia_tool, save_tool]

# Create the agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Enable verbose output for debugging
)

# Get user input for research query
user_query = input("what can I help you research?" + "\n" * 2)

# Execute the agent with the user's query
raw_response = agent_executor.invoke({"query": user_query})

# Print the raw output from the agent
print(f"raw output: {raw_response['output']}")

try:
    # Parse the raw output into a structured format
    structured_response = parser.parse(raw_response["output"])

    # Print the structured response
    print()
    print(f"structured response:")
    print(f"topic: {structured_response.topic}")
    print(f"summary: {structured_response.summary}")
    print(f"sources: {', '.join(structured_response.sources)}")
    print(f"tools_used: {', '.join(structured_response.tools_used)}")

except Exception as e:
    # Handle parsing errors
    print()
    print(f"note: could not parse structured response: {e}")
```

Where we have defined the tools in the `tools.py` file:

```python
import os
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

# Initialize the Wikipedia API wrapper with specified parameters
api_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=256,
)

# Create a tool for querying Wikipedia
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


def save_to_text(data: str, filename: str = "research-output"):
    # Check if the output directory exists, create if not
    if "research-outputs" not in os.listdir():
        os.makedirs("research-outputs")

    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the text to be saved
    formatted_text = f"""––– Research Output –––\nTimestamp: {timestamp}\n\n{data}\n\n"""
    filename_with_timestamp = f"{filename} - {timestamp}.txt"
    filepath = os.path.join("research-outputs", filename_with_timestamp)

    # Write the formatted text to the output file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Research data successfully saved to {filename_with_timestamp}"


# Define a tool for saving research output to a text file
save_tool = Tool(
    name="save_to_text",
    description="""Save structured research output to a text file. 
    Parameters:
        data: The research output to save
        filename: Name for the output file (without extension)
    """,
    func=save_to_text,
)
```