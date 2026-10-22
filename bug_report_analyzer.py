"""
AI Bug Report Analyzer (Gemini + LangGraph)

Workflow:
    START -> bug_analysis -> severity_detection -> root_cause_suggestion
          -> format_report -> END

Usage:
    python bug_report_analyzer.py "App crashes when I tap Pay on the checkout page"
    python bug_report_analyzer.py                 # interactive input
    python bug_report_analyzer.py --graph         # print the workflow as a Mermaid diagram
    python bug_report_analyzer.py "..." --out report.md
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()  # reads GOOGLE_API_KEY from a .env file or the environment

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


# --------------------------------------------------------------------------
# 1. Structured output schemas (the LLM must answer in these shapes)
# --------------------------------------------------------------------------
class BugAnalysis(BaseModel):
    title: str = Field(description="Short, specific bug title (max 12 words)")
    affected_feature: str = Field(description="The feature, screen or component affected")
    summary: str = Field(description="1-3 sentence summary of the issue")


class SeverityResult(BaseModel):
    severity: Literal["Critical", "High", "Medium", "Low"]
    reason: str = Field(description="One or two sentences justifying the severity")


class RootCauses(BaseModel):
    causes: List[str] = Field(description="3 to 5 plausible root causes, most likely first")


# --------------------------------------------------------------------------
# 2. Graph state shared by all nodes
# --------------------------------------------------------------------------
class BugState(TypedDict, total=False):
    bug_description: str
    title: str
    affected_feature: str
    summary: str
    severity: str
    severity_reason: str
    root_causes: List[str]
    final_report: str


# --------------------------------------------------------------------------
# 3. Nodes
# --------------------------------------------------------------------------
def bug_analysis(state: BugState) -> BugState:
    """Node 1: extract title, affected feature and summary."""
    chain = llm.with_structured_output(BugAnalysis)
    result = chain.invoke(
        "You are a QA engineer. Read the bug description and extract a title, "
        "the affected feature and a short summary. Use only facts from the text; "
        "do not invent details.\n\n"
        f"Bug description:\n{state['bug_description']}"
    )
    return {
        "title": result.title,
        "affected_feature": result.affected_feature,
        "summary": result.summary,
    }


def severity_detection(state: BugState) -> BugState:
    """Node 2: classify severity as Critical, High, Medium or Low."""
    chain = llm.with_structured_output(SeverityResult)
    result = chain.invoke(
        "Classify the severity of this bug using these rules:\n"
        "- Critical: crash, data loss, security issue or outage affecting many users, no workaround\n"
        "- High: a major feature is broken and there is no easy workaround\n"
        "- Medium: a feature is partly broken or a workaround exists\n"
        "- Low: cosmetic or minor issue with little user impact\n\n"
        f"Title: {state['title']}\n"
        f"Affected feature: {state['affected_feature']}\n"
        f"Summary: {state['summary']}\n"
        f"Original description: {state['bug_description']}"
    )
    return {"severity": result.severity, "severity_reason": result.reason}


def root_cause_suggestion(state: BugState) -> BugState:
    """Node 3: suggest possible causes."""
    chain = llm.with_structured_output(RootCauses)
    result = chain.invoke(
        "Suggest 3 to 5 possible root causes for this bug. They are hypotheses "
        "to investigate, not confirmed facts.\n\n"
        f"Title: {state['title']}\n"
        f"Affected feature: {state['affected_feature']}\n"
        f"Summary: {state['summary']}\n"
        f"Severity: {state['severity']}\n"
        f"Original description: {state['bug_description']}"
    )
    return {"root_causes": result.causes}


def format_report(state: BugState) -> BugState:
    """Node 4: build the final, readable bug report."""
    causes = "\n".join(f"{i}. {c}" for i, c in enumerate(state["root_causes"], 1))
    report = (
        f"# Bug Report: {state['title']}\n\n"
        f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"**Affected Feature:** {state['affected_feature']}\n\n"
        f"**Severity:** {state['severity']}\n"
        f"*Reason:* {state['severity_reason']}\n\n"
        f"## Summary\n{state['summary']}\n\n"
        f"## Possible Root Causes\n{causes}\n\n"
        f"## Original Description\n{state['bug_description']}\n"
    )
    return {"final_report": report}


# --------------------------------------------------------------------------
# 4. Build the graph
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(BugState)
    graph.add_node("bug_analysis", bug_analysis)
    graph.add_node("severity_detection", severity_detection)
    graph.add_node("root_cause_suggestion", root_cause_suggestion)
    graph.add_node("format_report", format_report)

    graph.add_edge(START, "bug_analysis")
    graph.add_edge("bug_analysis", "severity_detection")
    graph.add_edge("severity_detection", "root_cause_suggestion")
    graph.add_edge("root_cause_suggestion", "format_report")
    graph.add_edge("format_report", END)
    return graph.compile()


# --------------------------------------------------------------------------
# 5. Command-line interface
# --------------------------------------------------------------------------
def read_description(args) -> str:
    if args.description:
        return " ".join(args.description).strip()
    print("Describe the bug (press Enter on an empty line to finish):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip() and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description="AI Bug Report Analyzer")
    parser.add_argument("description", nargs="*", help="Bug description in plain English")
    parser.add_argument("--graph", action="store_true", help="Print the workflow as Mermaid")
    parser.add_argument("--out", help="Save the report to this file (e.g. report.md)")
    args = parser.parse_args()

    app = build_graph()

    if args.graph:
        print(app.get_graph().draw_mermaid())
        return

    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("Error: set GOOGLE_API_KEY in your environment or in a .env file.")

    description = read_description(args)
    if len(description) < 10:
        sys.exit("Error: please enter a bug description of at least a few words.")

    try:
        result = app.invoke({"bug_description": description})
    except Exception as exc:  # network, quota or API-key problems
        sys.exit(f"Error while calling Gemini: {exc}")

    print("\n" + result["final_report"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
