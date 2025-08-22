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
# -

# Imports essential modules from CrewAI.

# +
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# Access the variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = gemini_api_key

gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0,
)

# Run Local LLM to escape API quota limitations
llm = LLM(model="ollama/tinyllama")
# -

# Declaring the agents and tasks

# Create an agent with code execution enabled
analysis_agent = Agent(
    role="mathematician",
    goal="Analyze data and provide insights.",
    backstory="You are an experienced mathematician with experience in statistics.",
    verbose=True,
    llm=gemini_llm,
    max_iter=3,
)

# Create a task that requires code execution
data_analysis_task = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent,
    expected_output="Provide the dataset first and thent the the average age of the participants."
)

# Creating the crew and datasets list before commencing execution

# +
# Create a crew and add the task
analysis_crew = Crew(
    agents=[analysis_agent],
    tasks=[data_analysis_task]
)

# List of datasets to analyze
datasets = [
  { "ages": [25, 30, 35, 40, 45] },
  { "ages": [20, 25, 30, 35, 40] },
  { "ages": [30, 35, 40, 45, 50] }
]

result = analysis_crew.kickoff_for_each(inputs=datasets)

# +
from IPython.display import Markdown

for crew_output in result:
    result_markdown = crew_output.raw
    display(Markdown(result_markdown))
# -

# ### In the above output you can see that we are getting differently formatted output for each different execution instance. This is due to us not clarifying the exact format we wanted.
