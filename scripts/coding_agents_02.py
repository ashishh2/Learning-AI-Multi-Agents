# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

# Load environment variables from .env file
load_dotenv()

# Access the variables
openai_api_key = os.getenv("OPENAI_API_KEY")
# -

# Imports essential modules from CrewAI.

# +
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

os.environ["OPENAI_API_KEY"] = openai_api_key

# Run Local LLM to escape API quota limitations
llm = LLM(model="ollama/tinyllama")
# -

# Creates a crew and then kicks off the project.

# Create a debugging agent with code execution enabled
debugging_agent = Agent(
    role="Python Debugger",
    goal="Identify and fix issues in existing Python code",
    backstory="You are an experienced Python developer with a knack for finding and fixing bugs.",
    allow_code_execution=True,
    verbose=True,
    llm=llm,
    max_iter=3
)

# Define a task that involves debugging the provided code
debug_task = Task(
    description=(
        "The following Python code is supposed to return the square of each number in the list, "
        "but it contains a bug. Please identify and fix the bug:\n"
        "```\n"
        "numbers = [2, 4, 6, 8]\n"
        "squared_numbers = [n*m for n in numbers]\n"
        "print(squared_numbers)\n"
        "```"
    ),
    agent=debugging_agent,
    expected_output="The corrected code should output the squares of the numbers in the list. Provide the updated code and tell what was the bug and how you fixed it."
)

# +
# Form a crew and assign the debugging task
debug_crew = Crew(
    agents=[debugging_agent],
    tasks=[debug_task]
)

# Execute the crew and retrieve the result
result = debug_crew.kickoff()

# +
from IPython.display import Markdown

# Convert the CrewOutput object to a Markdown string
result_markdown = result.raw

# Display the result as Markdown
Markdown(result_markdown)
# -


