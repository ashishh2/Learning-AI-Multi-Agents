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

# Run Local LLM to escape API rate limits
llm = LLM(model="ollama/tinyllama")
# -

# Declaring the agents and tasks

# +
# Create agents
analysis_agent1 = Agent(
    role="Mathematician",
    goal="Analyze data and provide insights.",
    backstory="You are an experienced mathematician with experience in statistics.",
    verbose=True,
    llm=llm,
    max_iter=3
)

analysis_agent2 = Agent(
    role="Mathematician",
    goal="Analyze data and provide insights.",
    backstory="You are an experienced mathematician with experience in statistics.",
    verbose=True,
    llm=llm,
    max_iter=3
)

# +
# Create tasks
data_analysis_task1 = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent1,
    expected_output="Provide the dataset first and thent the the average age of the participants.",
    async_execution=True
)

# Create a task that requires code execution
data_analysis_task2 = Task(
    description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
    agent=analysis_agent2,
    expected_output="Provide the dataset first and thent the the average age of the participants."
)
# -

# Creating the crew and datasets list before commencing execution

# +
# Create a crew and add the task
analysis_crew1 = Crew(
    agents=[analysis_agent1],
    tasks=[data_analysis_task1]
)

analysis_crew2 = Crew(
    agents=[analysis_agent2],
    tasks=[data_analysis_task2]
)

result_1 = await analysis_crew1.kickoff_async(inputs={"ages": [25, 30, 35, 40, 45]})
result_2 = analysis_crew2.kickoff(inputs={"ages": [20, 25, 30, 35, 40]})
# -

print("Async Crew Thread Output:", result_1)
print("Main Thread Output", result_2)
