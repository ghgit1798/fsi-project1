from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.router import router_agent
from agents.specialists import (
    account_security_agent,
    ai_security_agent,
    device_security_agent,
    general_security_agent,
)


class ChatState(TypedDict, total=False):
    question: str
    route: str
    answer: str


def router_node(state: ChatState) -> ChatState:
    return {"route": router_agent(state["question"])}


def ai_security_node(state: ChatState) -> ChatState:
    return {"answer": ai_security_agent(state["question"])}


def device_security_node(state: ChatState) -> ChatState:
    return {"answer": device_security_agent(state["question"])}


def account_security_node(state: ChatState) -> ChatState:
    return {"answer": account_security_agent(state["question"])}


def general_security_node(state: ChatState) -> ChatState:
    return {"answer": general_security_agent(state["question"])}


def route_by_router(state: ChatState) -> str:
    return state.get("route", "general_security")


builder = StateGraph(ChatState)
builder.add_node("router", router_node)
builder.add_node("ai_security", ai_security_node)
builder.add_node("device_security", device_security_node)
builder.add_node("account_security", account_security_node)
builder.add_node("general_security", general_security_node)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_by_router,
    {
        "ai_security": "ai_security",
        "device_security": "device_security",
        "account_security": "account_security",
        "general_security": "general_security",
    },
)
builder.add_edge("ai_security", END)
builder.add_edge("device_security", END)
builder.add_edge("account_security", END)
builder.add_edge("general_security", END)
security_graph = builder.compile()


def run_security_workflow(question: str) -> tuple[str, str]:
    result = security_graph.invoke({"question": question})
    route = result.get("route", "general_security")
    answer = result.get("answer", "답변을 생성하지 못했습니다. 다시 시도해 주세요.")
    return route, answer
