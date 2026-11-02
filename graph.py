## graph.py - connects the 3 agents into one workflow using LangGraph (state graph)
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from agents import planner_agent, researcher_agent, writer_agent


class AgentState(TypedDict, total=False):
    ## this is the shared memory that every agent reads from and writes to
    query: str
    history: list      ## past (question, answer) pairs -> short-term memory
    is_financial: bool
    plan: list
    data: list         ## facts collected by tools
    answer: str
    feedback: str      ## what the Writer says is missing (used for retry)
    retries: int
    trace: Annotated[list, operator.add]  ## operator.add means new trace lines get appended


def after_planner(state):
    ## if the question is off-topic, finish; otherwise go to the researcher
    return "researcher" if state["is_financial"] else END


def after_researcher(state):
    ## if no facts were collected, show the Researcher's honest fallback instead of asking the Writer to guess
    return "writer" if state.get("data") else END


def after_writer(state):
    ## if the writer asked for more data, loop back; otherwise finish
    return "researcher" if state.get("feedback") else END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_agent)
    g.add_node("researcher", researcher_agent)
    g.add_node("writer", writer_agent)

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", after_planner)
    g.add_conditional_edges("researcher", after_researcher)
    g.add_conditional_edges("writer", after_writer)
    return g.compile()


def new_state(query, history):
    ## starting values for every new question
    return {"query": query, "history": history, "data": [], "retries": 0, "feedback": "", "trace": []}


def run_query(graph, query, history=None):
    ## used by the terminal test script
    if not query.strip():  ## edge case: empty input
        return {"answer": "Please type a financial question.", "trace": ["[System] Empty input."]}
    return graph.invoke(new_state(query, history or []))
