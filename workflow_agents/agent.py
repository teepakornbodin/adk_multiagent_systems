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

# Import Wikipedia Tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Setup Logging
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")
print(f"Using Model: {model_name}")

# ==========================================
# 1. TOOLS DEFINITION
# ==========================================

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key (pos_data, neg_data, etc.)."""
    existing_state = tool_context.state.get(field, [])
    # ตรวจสอบว่า existing_state เป็น list หรือไม่ ถ้าไม่ใช่ให้แปลงเป็น list
    if isinstance(existing_state, str):
        existing_state = [existing_state]
        
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}

def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """Write the final verdict to a text file."""
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "success"}

# Wikipedia Tool Setup
wiki_tool = LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))

# ==========================================
# 2. AGENTS DEFINITION (The Historical Court)
# ==========================================

# --- Step 2: The Investigation (Parallel Agents) ---

# Agent A: The Admirer (หาข้อมูลด้านบวก)
admirer_agent = Agent(
    name="admirer_agent",
    model=model_name,
    description="Researches positive achievements and successes.",
    instruction="""
    ROLE: You are 'The Admirer'. Your job is to find ONLY positive information, achievements, and virtues about the subject.

    SUBJECT: { PROMPT? }
    CURRENT JUDGE FEEDBACK: { judge_feedback? }
    
    INSTRUCTIONS:
    - If 'judge_feedback' asks for specific positive details, search for those.
    - Otherwise, search for the greatest achievements and legacy of the SUBJECT.
    - Use the 'wikipedia' tool to find facts.
    - Use 'append_to_state' to save your findings to the field 'pos_data'.
    - Keep your summary brief and focused on the good side.
    """,
    tools=[wiki_tool, append_to_state]
)

# Agent B: The Critic (หาข้อมูลด้านลบ/ข้อโต้แย้ง)
critic_agent = Agent(
    name="critic_agent",
    model=model_name,
    description="Researches controversies, failures, and criticisms.",
    instruction="""
    ROLE: You are 'The Critic'. Your job is to find ONLY negative information, controversies, crimes, or failures about the subject.

    SUBJECT: { PROMPT? }
    CURRENT JUDGE FEEDBACK: { judge_feedback? }

    INSTRUCTIONS:
    - If 'judge_feedback' asks for specific negative details, search for those.
    - Otherwise, search for "controversy", "criticism", "failures", or "war crimes" of the SUBJECT.
    - Use the 'wikipedia' tool to find facts.
    - Use 'append_to_state' to save your findings to the field 'neg_data'.
    - Keep your summary brief and focused on the bad side.
    """,
    tools=[wiki_tool, append_to_state]
)

# Group Step 2 into Parallel Execution
investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer_agent, critic_agent]
)

# --- Step 3: The Trial & Review (The Judge) ---

# Agent C: The Judge (ตรวจสอบความสมดุลและสั่งงานต่อ)
judge_agent = Agent(
    name="judge_agent",
    model=model_name,
    description="Reviews the evidence and decides if more research is needed.",
    instruction="""
    ROLE: You are 'The Judge'. You review the evidence collected by the Admirer and the Critic.

    EVIDENCE FOR PROSECUTION (NEGATIVE):
    { neg_data? }

    EVIDENCE FOR DEFENSE (POSITIVE):
    { pos_data? }

    INSTRUCTIONS:
    1. Analyze the quantity and quality of the information in 'neg_data' and 'pos_data'.
    2. DECISION LOGIC:
       - If EITHER side has too little information or the arguments are weak/unbalanced:
         -> Use 'append_to_state' to write specific instructions to 'judge_feedback' (e.g., "Critic, find more details on the 1990 scandal").
         -> Do NOT exit the loop.
       - If BOTH sides have sufficient and balanced information to form a verdict:
         -> Use the 'exit_loop' tool to end the trial.
    """,
    tools=[append_to_state, exit_loop]
)

# Group Step 2 & 3 into a Loop (The Trial Loop)
trial_process = LoopAgent(
    name="trial_process",
    description="The loop of investigation and judicial review.",
    sub_agents=[
        investigation_team, # Parallel search
        judge_agent         # Check results
    ],
    max_iterations=4  # Limit loops to prevent infinite running
)

# --- Step 4: The Verdict (Output) ---

verdict_writer = Agent(
    name="verdict_writer",
    model=model_name,
    description="Writes the final neutral report.",
    instruction="""
    ROLE: You are the Court Clerk writing the final Verdict.

    SUBJECT: { PROMPT? }
    POSITIVE EVIDENCE: { pos_data? }
    NEGATIVE EVIDENCE: { neg_data? }

    INSTRUCTIONS:
    - Write a comprehensive, NEUTRAL report comparing the facts.
    - Start with an introduction of the subject.
    - Present the arguments from the Admirer (Achievements).
    - Present the arguments from the Critic (Controversies).
    - Conclude with a balanced verdict summarizing their historical impact.
    - Use the 'write_file' tool to save this report:
        - directory: "court_records"
        - filename: "{PROMPT}.txt" (remove spaces for filename)
        - content: The full report.
    """,
    tools=[write_file]
)

# ==========================================
# 3. ROOT AGENT (Entry Point)
# ==========================================

# Step 1: The Inquiry (Sequential Wrapper)
# รับชื่อจาก User -> เข้าสู่กระบวนการศาล -> เขียนรายงาน
root_agent = Agent(
    name="historical_court_clerk",
    model=model_name,
    description="Starts the Historical Court session.",
    instruction="""
    - Greet the user and welcome them to 'The Historical Court'.
    - Ask the user for the name of a historical figure or event they want to put on trial.
    - Once the user provides a name, use 'append_to_state' to save it to 'PROMPT'.
    - Then, handover to the 'court_system'.
    """,
    tools=[append_to_state],
    sub_agents=[
        SequentialAgent(
            name="court_system",
            sub_agents=[
                trial_process,  # Loop (Investigate -> Judge)
                verdict_writer  # Output
            ]
        )
    ]
)