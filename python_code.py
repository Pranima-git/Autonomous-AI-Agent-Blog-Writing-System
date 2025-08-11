from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load .env keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
print("🔐 API Key Loaded:", groq_api_key)

# Set up the LLM (LLaMA3 via Groq)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="groq/llama3-8b-8192"  
)


# Agent to create outline
outline_agent = Agent(
    role="Content Planner",
    goal="Create a clear blog outline",
    backstory="Expert at structuring blog posts for students",
    verbose=True,
    llm=llm
)

outline_task = Task(
    description="Create a detailed outline for a blog titled 'Top 5 AI Tools for Students in 2025'.",
    agent=outline_agent,
    expected_output="Bullet points for each section."
)

# Agent to write the blog
writer_agent = Agent(
    role="Blog Writer",
    goal="Write engaging blog content",
    backstory="Skilled blog writer with a clear, friendly tone.",
    verbose=True,
    llm=llm
)

write_task = Task(
    description="Based on the outline, write a 1000+ word blog article.",
    agent=writer_agent,
    expected_output="Complete blog post with headings and subheadings."
)

# Crew setup
crew = Crew(
    agents=[outline_agent, writer_agent],
    tasks=[outline_task, write_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()

print("\n📄 FINAL BLOG OUTPUT:\n")
print(result)
