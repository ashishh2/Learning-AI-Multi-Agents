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

# +
from crewai import Agent, Task, Crew

# Create an agent with code execution enabled
coding_agent = Agent(
    role="Python Data Analyst",
    goal="Write and execute Python code to perform calculations",
    backstory="You are an experienced Python developer, skilled at writing efficient code to solve problems.",
    allow_code_execution=True,
    llm=llm,
    max_iter=3
)

# Define the task with explicit instructions to generate and execute Python code
data_analysis_task = Task(
    description=(
        "Write Python code to calculate the average of the following list of ages: [23, 35, 31, 29, 40]. "
        "Output the result in the format: 'The average age of participants is: <calculated_average_age>'"
    ),
    agent=coding_agent,
    expected_output="The generated code based on the requirments and the average age of participants is: <calculated_average_age>."
)

# Create a crew and add the task
analysis_crew = Crew(
    agents=[coding_agent],
    tasks=[data_analysis_task]
)

# Execute the crew
result = analysis_crew.kickoff()

print(result)

# +
from IPython.display import Markdown

# Convert the CrewOutput object to a Markdown string
result_markdown = result.raw

# Display the result as Markdown
Markdown(result_markdown)
