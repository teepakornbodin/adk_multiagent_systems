import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.genai import types
from google.adk.tools import exit_loop

# Wikipedia utilities
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# Initialize Cloud Logging
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

# Load environment variables
load_dotenv()

model_name = os.getenv("MODEL")
print(f"Using Model: {model_name}")


# =====================================================
# SECTION 1 — TOOL FUNCTIONS
# =====================================================

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """
    Store new information into a list-based state field
    such as pos_data or neg_data.
    """
    existing_state = tool_context.state.get(field, [])

    # Ensure the state value is always treated as a list
    if isinstance(existing_state, str):
        existing_state = [existing_state]

    tool_context.state[field] = existing_state + [response]
    logging.info(f"[State Updated → {field}] {response}")
    return {"status": "success"}


def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """
    Save the generated final report into a text file.
    """
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {"status": "success"}


# Prepare Wikipedia search tool
wiki_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)


# =====================================================
# SECTION 2 — AGENT DEFINITIONS
# =====================================================

# -------------------------------
# Investigation Stage (Parallel)
# -------------------------------

# Agent responsible for collecting positive facts
admirer_agent = Agent(
    name="admirer_agent",
    model=model_name,
    description="Collects achievements and positive historical contributions.",
    instruction="""
    ROLE: You are 'The Admirer'. Focus ONLY on positive aspects, achievements, and contributions.

    SUBJECT: { PROMPT? }
    CURRENT JUDGE FEEDBACK: { judge_feedback? }

    GUIDELINES:
    - If the Judge provides feedback, refine your search accordingly.
    - Otherwise, gather key achievements and long-term contributions.
    - Use the Wikipedia tool for reliable information.
    - Save results into 'pos_data' using append_to_state.
    - Keep responses concise and positive.
    """,
    tools=[wiki_tool, append_to_state]
)


# Agent responsible for collecting negative/critical facts
critic_agent = Agent(
    name="critic_agent",
    model=model_name,
    description="Collects controversies, criticisms, and negative history.",
    instruction="""
    ROLE: You are 'The Critic'. Focus ONLY on controversies, failures, and negative aspects.

    SUBJECT: { PROMPT? }
    CURRENT JUDGE FEEDBACK: { judge_feedback? }

    GUIDELINES:
    - If the Judge provides feedback, refine your search accordingly.
    - Otherwise, search for controversies, failures, crimes, or criticisms.
    - Use the Wikipedia tool for factual data.
    - Save results into 'neg_data' using append_to_state.
    - Keep responses concise and critical-focused.
    """,
    tools=[wiki_tool, append_to_state]
)


# Run both research agents at the same time
investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer_agent, critic_agent]
)


# -------------------------------
# Review Stage (Judge + Loop)
# -------------------------------

judge_agent = Agent(
    name="judge_agent",
    model=model_name,
    description="Evaluates balance and completeness of collected evidence.",
    instruction="""
    ROLE: You are 'The Judge'. Review the evidence gathered from both sides.

    NEGATIVE SIDE:
    { neg_data? }

    POSITIVE SIDE:
    { pos_data? }

    TASK:
    1. Compare both sides in terms of completeness and balance.
    2. DECISION:
       - If one side lacks sufficient evidence:
         -> Provide targeted feedback via 'judge_feedback'.
         -> Continue the loop (do NOT exit).
       - If both sides are sufficiently detailed and balanced:
         -> Use 'exit_loop' to finish the trial phase.
    """,
    tools=[append_to_state, exit_loop]
)


# Loop combines investigation and evaluation until completion
trial_process = LoopAgent(
    name="trial_process",
    description="Repeats investigation and review until balanced.",
    sub_agents=[
        investigation_team,
        judge_agent
    ],
    max_iterations=4  # Safety limit to prevent endless looping
)


# -------------------------------
# Final Report Generation
# -------------------------------

verdict_writer = Agent(
    name="verdict_writer",
    model=model_name,
    description="Produces a balanced final historical report.",
    instruction="""
    ROLE: Court Clerk responsible for writing the final verdict.

    SUBJECT: { PROMPT? }
    POSITIVE EVIDENCE: { pos_data? }
    NEGATIVE EVIDENCE: { neg_data? }

    REQUIREMENTS:
    - Produce a neutral and balanced historical report.
    - Begin with an overview of the subject.
    - Present positive contributions.
    - Present criticisms and controversies.
    - Conclude with an objective summary of historical impact.
    - Save the report using 'write_file':
        directory: "court_records"
        filename: "{PROMPT}.txt"
        content: full report
    """,
    tools=[write_file]
)


# =====================================================
# SECTION 3 — ROOT AGENT (ENTRY POINT)
# =====================================================

root_agent = Agent(
    name="historical_court_clerk",
    model=model_name,
    description="Initializes and manages the Historical Court session.",
    instruction="""
    - Welcome the user to the Historical Court.
    - Ask for a historical figure or event to analyze.
    - Store the provided subject into 'PROMPT'.
    - Then proceed to the full court process.
    """,
    tools=[append_to_state],
    sub_agents=[
        SequentialAgent(
            name="court_system",
            sub_agents=[
                trial_process,
                verdict_writer
            ]
        )
    ]
)
