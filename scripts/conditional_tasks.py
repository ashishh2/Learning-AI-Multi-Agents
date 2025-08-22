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
#     display_name: Project Env
#     language: python
#     name: project_env
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
from crewai_tools import TavilySearchTool

# Access the variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["GEMINI_API_KEY"] = gemini_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0,
)

# Run Local LLM to escape API quota limitations
llm = LLM(model="ollama/tinyllama")

# Initialize TavilySearchTool
search_tool = TavilySearchTool()
# -

# Sets up the conditional task checker

# +
from typing import List
from pydantic import BaseModel
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput

# Define a function to assess whether the data needs to be augmented
def should_fetch_more_data(output: TaskOutput) -> bool:
    return len(output.pydantic.events) < 3  # Condition to trigger task


# -

# Agents declared

# +
# Create agents for different roles
data_collector = Agent(
    role="Data Collector",
    goal="Retrieve event data using Tavily Search Tool",
    backstory="You have a knack for finding the most exciting events happening around.",
    verbose=True,
    tools=[search_tool],
    llm=gemini_llm,
    max_iter=3,
)

data_analyzer = Agent(
    role="Data Analyzer",
    goal="Analyze the collected data",
    backstory="You're known for your analytical skills, making sense of complex datasets.",
    verbose=True,
    llm=gemini_llm,
    max_iter=3,
)

summary_creator = Agent(
    role="Summary Creator",
    goal="Produce a concise summary from the event data",
    backstory="You're a skilled writer, able to summarize information clearly and effectively.",
    verbose=True,
    llm=gemini_llm,
    max_iter=3,
)


# -

# Tasks are declared as we did before however the conditional task for this example is declared with ConditionalTask() instead

# +
class EventsData(BaseModel):
    events: List[str]

# Define the tasks
fetch_task = Task(
    description="Collect event data for Bangalore, India using Serper tool",
    expected_output="A list of 3 exciting events happening in Bangalore, India this week",
    agent=data_collector,
    output_pydantic=EventsData,
)

verify_data_task = ConditionalTask(
    description="""
        Ensure that sufficient event data has been collected.
        If fewer than 3 events are found, gather more using the Serper tool.
        """,
    expected_output="An updated list of at least 3 events happening in Bangalore, India this week",
    condition=should_fetch_more_data,
    agent=data_analyzer,
)

summary_task = Task(
    description="Summarize the collected events data for Bangalore, India",
    expected_output="summary_generated",
    agent=summary_creator,
)

# +
# Assemble the crew with the defined agents and tasks
crew = Crew(
    agents=[data_collector, data_analyzer, summary_creator],
    tasks=[fetch_task, verify_data_task, summary_task],
    verbose=True,
    planning=True,  # Retain the planning feature
    planning_llm=gemini_llm,
)

# Execute the tasks with the crew
result = crew.kickoff()
print("results", result)

# +
from IPython.display import Markdown

# Convert the CrewOutput object to a Markdown string
result_markdown = result.raw

# Display the result as Markdown
Markdown(result_markdown)
# -


