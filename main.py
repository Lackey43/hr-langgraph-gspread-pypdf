from typing import Literal, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os


load_dotenv()

from agents.agents import research_agent, therapist_agent, marketing_agent
from model.models import google_model

config = {"configurable": {"thread_id": "1"}}

# Schema for structured output
class Route(BaseModel):
    step: Literal["research", "therapy", "marketing"] = Field(description="Check if query is for research, therapy or marketing")

router = google_model.with_structured_output(Route)

# State
class State(TypedDict):
    query: str
    decision: str
    output: str
    messages: Annotated[list, "add_messages"]

# Nodes
def researcher_agent(state: State):
    """Research Agent"""
    result = research_agent.invoke(state["query"])
    print(result)

    return {
        "messages": result
    }


def therapy_agent(state: State):
    """Therapy Agent"""
    result = therapist_agent.invoke(state["query"])
    print(result)

    return {
        "messages": result
    }

def marketer_agent(state: State):
    """Marketing Agent"""
    result = marketing_agent.invoke(state["query"])
    print(result)

    return {
        "messages": result
    }

def router_agent(state: State):
    """Route the query"""
    decision = router.invoke([
        SystemMessage(content=f"Route the query to research, therapy, or marketing based on the user's request."),
        HumanMessage(f"This is my query:\n{state["query"]}")
    ])
    print(decision)
    return {"decision": decision.step}

def route_decision(state: State):
    if state["decision"]== "research":
        return "researcher_agent"
    elif state["decision"] == "therapy":
        return "therapy_agent"
    elif state["decision"] == "marketing":
        return "marketer_agent"

# Build Graph
router_builder = StateGraph(State)

router_builder.add_node("router_agent", router_agent)
router_builder.add_node("researcher_agent", researcher_agent)
router_builder.add_node("therapy_agent", therapy_agent)
router_builder.add_node("marketer_agent", marketer_agent)

router_builder.add_edge(START, "router_agent")

router_builder.add_conditional_edges(
    "router_agent",
    route_decision,
    {
        "researcher_agent": "researcher_agent",
        "therapy_agent": "therapy_agent",
        "marketer_agent": "marketer_agent",
    }
)

router_builder.add_edge("researcher_agent", END)
router_builder.add_edge("therapy_agent", END)
router_builder.add_edge("marketer_agent", END)

# Compile
memory = MemorySaver()
router_workflow = router_builder.compile(checkpointer=memory)

print(router_workflow.get_graph().draw_ascii())

# Run
if __name__ == "__main__":
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
            
        result = router_workflow.invoke({"query": query}, config=config)
        print("\nResponse:", result.get("output", "No output"))