import os

from langchain.agents import Tool
from crewai import Agent, Task, Process, Crew
from langchain.utilities import GoogleSerperAPIWrapper


search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="useful for when you need to ask the agent to search the internet",
)

# To Load GPT-4
api = os.environ.get("OPENAI_API_KEY")

explorer = Agent(
    role="Senior Researcher",
    goal="Find and explore the newest industry news and peer-reviewed academic research covering AI, genAI, crypto, blockchain embedded insurance, and new business models in the insurance space (with a focus on life insurance)",
    backstory="""You are an Expert Academic Researcher with a deep knowledge of insurance (particularly life insurance) that knows how to find and comb the academic journals, forums, LinkedIn posts, etc. to spot emerging technology trends and published research in insurance. ONLY use scraped data from the internet for the report.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
)

writer = Agent(
    role="Senior Technical Writer",
    goal="Write succinct summaries of technical content",
    backstory="""You are an Expert Writer on technical innovation, especially in the field of AI and machine learning. You know how to write in an engaging, interesting but simple, straightforward, and concise manner. You know how to present complicated technical terms to a general audience in a clear and engaging way. ONLY use scraped data from the internet. You will review the Senior Researcher’s findings and draft an email update that summarizes each item found and provides a link (in a list format).""",
    verbose=True,
    allow_delegation=True,
)

critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback and criticize research and content. Make sure that any research identified is of high academic quality and that the key issues have been identified in any summary. Ensure the writing tone and writing style is compelling, simple, and concise",
    backstory="""You are an Expert Academic Professor who knows insurance deeply, and you are providing feedback to the Senior Researcher and Writer. You can tell when research is not of good quality, and you can tell what leading edge technologies are worth identifying and writing about. You know how to provide helpful feedback that can improve any text. You know how to make sure that text stays technical and insightful while using layman terms where necessary.""",
    verbose=True,
    allow_delegation=False,
)

task1 = Task(
    description=""""
    find and explore the newest industry news and peer-reviewed academic research covering AI, genAI, crypto, blockchain embedded insurance, and new business models in the insurance space (with a focus on life insurance).
Identify key trends, breakthrough technologies, and potential industry impacts.""",
    expected_output="""Full analysis report in bullet points with links to sources""",
    agent=explorer,
)

task2 = Task(
    description="""Using the insights provided and draft an email update that summarises each item found and provides a link (in a list format).
  Your writing should be informative  catering to a tech-savvy academic audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
    """,
    expected_output="""
    Full blog style post of at least 4 paragraphs""",
    agent=writer,
)

task3 = Task(
    description="""Critique and suggest improvements to the output of the senior technical writer.
     Make sure that there is academic research (peer-reviewed from reputable sources) and implications for industry, and that the info is leading edge with a focus on embedded insurance, AI, and life insuranc
    """,
    expected_output="""Comments and suggested changes""",
    agent=critic,
)

crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task1, task2, task3],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

result = crew.kickoff()
print(result)
