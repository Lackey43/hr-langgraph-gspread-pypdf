from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from agents.agents import hr_agent
from model.models import google_model
from tools.pdf_extractor import extract_pdf
from langchain.messages import SystemMessage, HumanMessage
import json
from langchain_core.prompts import ChatPromptTemplate
from tools.merm import outputImage
import operator

class PdfExtractor(BaseModel):
    name: str = Field(description="Name of Applicant")
    age: str = Field(description="age of Applicant")
    contact: str = Field(description="contact number of applicant")
    email: str = Field(description="email address of Applicant")
    summary: str = Field(
        description="summary of all work experiences, skills and certificates of the applicant"
    )
    skills : list = Field(description="list of skills of the applicant")
    work_experience : list = Field(description=" list of work experience of the applicant")
    certificates : list = Field(description=" list of certificates of the applicant")

class Recommendation(BaseModel):
    rate : int = Field(description="this is the rating from 1 to 10 you will give the applicant of how qualified he is for the role")
    recom: str = Field(description="this is the assessment or feedback you want to give")



class State(TypedDict):
    query: str
    output: Annotated[list, operator.add]
    extracted: str



structured_ai = google_model.with_structured_output(PdfExtractor)
structured_ai2= google_model.with_structured_output(Recommendation)

def parse_pdf(state: State):



    message =structured_ai.invoke(
                [SystemMessage(f'You are gonna extract the data found here:\n{state["extracted"]}'),
                HumanMessage("start extracting")
                ]


        )
    print(message)
    return {"output": [message]}



def show_result(state: State):
    message =structured_ai2.invoke(
            [SystemMessage(
                """
                You are an expert HR Resume Analyst and Optimizer with 10+ years of experience in
                talent acquisition and recruitment for tech, HR, and administrative roles.
                """
                ),
            HumanMessage(f'This is the role: {state["query"]}\n\nthis is the applicant\'s background:\n{state["output"]}')
                ,
            ]

    )
    print(message)
    return {"output" : [message]}



graph = StateGraph(State)
graph.add_node("test1", parse_pdf)
graph.add_node("test", show_result)
graph.add_edge(START, "test1")
graph.add_edge("test1", "test")
graph.add_edge("test", END)

agent = graph.compile()


# outputImage(agent)


if __name__ == "__main__":
    extracted = extract_pdf("claude_daigan_resume.pdf")

    query = "english teacher"
    result = agent.invoke({"query": query, "extracted" : extracted})
    print("\n\n")
    print(result["output"])
    with open("output.json", "w") as f:
        # json.dump((f.model_dump(mode="json") for f in result["output"]),f, indent=2)
        json.dump([x.model_dump(mode="json") for x in result["output"]],f, indent=2)
    # with open("output.json", "w", encoding="utf-8") as f:
    #     json.dump(result,f, ensure_ascii=0, indent=2)
