from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

# Define an LLM instance outside the CrewBase class.
llm = LLM(model="groq/llama3-8b-8192", temperature=0)

# Define the knowledge source from the Pakistan Post website outside the class.
content_source = CrewDoclingSource(
    file_paths=[
        "https://www.pakpost.gov.pk"
    ],
)

@CrewBase
class DevCrew:

    @agent
    def pakistan_post_agent(self) -> Agent:
        return Agent(
            role="Pakistan Post Service Agent",
            goal="Answer queries about Pakistan Post services using data from the Pakistan Post website.",
            backstory="You are an expert on Pakistan Post services with in-depth knowledge of the website's content.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    @agent
    def roman_urdu_translator(self) -> Agent:
        return Agent(
            role="Roman Urdu Translator",
            goal="Translate the provided English text into Roman Urdu (using English letters) accurately.",
            backstory="You are skilled in accurately translating English text into Roman Urdu.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    @task
    def answer_pakpost_query(self) -> Task:
        return Task(
            description="Answer the following question using information from the Pakistan Post website: {question}",
            expected_output="A concise, accurate answer referencing the Pakistan Post website with sources.",
            agent=self.pakistan_post_agent()
        )

    @task
    def translate_to_roman_urdu(self) -> Task:
        return Task(
            description="Translate the following English text into Roman Urdu (using English letters): {{answer}}",
            expected_output="The text translated into Roman Urdu using English script.",
            agent=self.roman_urdu_translator()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[content_source]
        )
