"""Offline tests: run the whole graph with a fake LLM, so no API key is needed."""
import agents
from workflow import run_workflow

calls = []


def fake_llm(system_prompt: str, message: str, max_tokens: int = 400) -> str:
    calls.append(system_prompt[:40])
    if "gatekeeper" in system_prompt:
        return "Other" if "capital" in message else "Feedback"
    if "sentiment" in system_prompt:
        return "Negative" if "late" in message else "Positive"
    return "FAKE REPLY"


agents.ask_llm = fake_llm


def check(message: str, scope: str, sentiment: str, n_calls: int) -> None:
    calls.clear()
    result = run_workflow(message)
    assert result["scope"] == scope, result
    assert result["sentiment"] == sentiment, result
    assert len(calls) == n_calls, calls
    print(f"PASS  {message!r} -> {scope}, {sentiment}, {n_calls} LLM call(s)")


check("I love your service! The staff were helpful.", "Feedback", "Positive", 3)
check("I hate your services. The delivery was late.", "Feedback", "Negative", 3)
check("What is the capital of France?", "Other", "Not applicable", 1)
assert run_workflow("What is the capital of France?")["response"] == agents.OFF_TOPIC_REPLY
assert agents.pick_label(" Answer: positive. ", ("Positive", "Negative")) == "Positive"
assert agents.pick_label("**Other**", ("Feedback", "Other")) == "Other"
try:
    agents.pick_label("Positive, I think", ("Positive", "Negative"))
except ValueError:
    print("PASS  an unclear label raises ValueError")
else:
    raise AssertionError("pick_label accepted an unclear label")
print("All tests passed.")
