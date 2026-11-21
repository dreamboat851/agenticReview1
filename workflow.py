"""LangGraph routing for the four-agent customer feedback application."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agents import (
    OFF_TOPIC_REPLY,
    gatekeeper_agent,
    sentiment_agent,
    positive_agent,
    negative_agent,
)


class FeedbackState(TypedDict):
    feedback: str
    scope: str
    sentiment: str
    response: str


def check_scope(state: FeedbackState) -> dict:
    return {"scope": gatekeeper_agent(state["feedback"])}


def reply_off_topic(state: FeedbackState) -> dict:
    return {"sentiment": "Not applicable", "response": OFF_TOPIC_REPLY}


def classify(state: FeedbackState) -> dict:
    return {"sentiment": sentiment_agent(state["feedback"])}


def respond_positive(state: FeedbackState) -> dict:
    return {"response": positive_agent(state["feedback"])}


def respond_negative(state: FeedbackState) -> dict:
    return {"response": negative_agent(state["feedback"])}


def route_scope(state: FeedbackState) -> str:
    scope = state["scope"]
    if scope == "Feedback":
        return "in_scope"
    if scope == "Other":
        return "out_of_scope"
    raise ValueError(f"Cannot route unexpected scope: {scope!r}")


def route_sentiment(state: FeedbackState) -> str:
    sentiment = state["sentiment"]
    if sentiment == "Positive":
        return "positive"
    if sentiment == "Negative":
        return "negative"
    raise ValueError(f"Cannot route unexpected sentiment: {sentiment!r}")


builder = StateGraph(FeedbackState)
builder.add_node("gatekeeper", check_scope)
builder.add_node("off_topic", reply_off_topic)
builder.add_node("classify", classify)
builder.add_node("positive", respond_positive)
builder.add_node("negative", respond_negative)
builder.add_edge(START, "gatekeeper")
builder.add_conditional_edges(
    "gatekeeper", route_scope, {"in_scope": "classify", "out_of_scope": "off_topic"}
)
builder.add_conditional_edges(
    "classify", route_sentiment, {"positive": "positive", "negative": "negative"}
)
builder.add_edge("off_topic", END)
builder.add_edge("positive", END)
builder.add_edge("negative", END)
graph = builder.compile()


def run_workflow(feedback: str) -> FeedbackState:
    start = {"feedback": feedback, "scope": "", "sentiment": "", "response": ""}
    return graph.invoke(start)
