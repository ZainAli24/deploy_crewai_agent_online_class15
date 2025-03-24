#!/usr/bin/env python
from crewai.flow import Flow, start, listen
from deploy_crewai_agent_online_class15.crew import DevCrew

class DevFlow(Flow):
    @start()
    def get_user_input(self):
        # Get the question from the user
        self.state["question"] = input("Please enter your question: ")
    
    @listen(get_user_input)
    def run_crew(self):
        # Create an instance of your crew and kick it off with the user's question.
        crew_instance = DevCrew()
        result = crew_instance.crew().kickoff(
            inputs={"question": self.state["question"]}
        )
        self.state["answer"] = result.raw
        return result
    
    @listen(run_crew)
    def output_result(self, result):
        # Output the final answer from the crew.
        print("Final Answer:")
        print(self.state["answer"])
        return result

def main():
    flow = DevFlow()
    res = flow.kickoff()
    print(res)

