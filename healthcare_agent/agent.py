"""
healthcare_agent — Agent definition.

This agent has read-only access to a patient's FHIR R4 record.
FHIR credentials (server URL, bearer token, patient ID) are injected via the
A2A message metadata by the caller (e.g. Prompt Opinion) and extracted into
session state by extract_fhir_context before every LLM call.

To customise:
  • Change model, description, and instruction below.
  • Add or remove tools from the tools=[...] list.
  • Add new FHIR tools in shared/tools/fhir.py and export from shared/tools/__init__.py.
  • Add non-FHIR tools in shared/tools/ or locally in a tools/ folder here.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool

from healthcare_agent.agents.diagnosis_agent import diagnosis_agent
from healthcare_agent.agents.insurance_agent import insurance_agent
from healthcare_agent.agents.care_plan_agent import care_plan_agent
from healthcare_agent.agents.insurance_agent import insurance_agent 
from healthcare_agent.agents.lab_analysis_agent import lab_analysis_agent
from healthcare_agent.agents.medication_agent import medication_agent
from healthcare_agent.agents.triage_agent import triage_agent

from shared.fhir_hook import extract_fhir_context
from shared.tools import (
    get_active_conditions,
    get_active_medications,
    get_patient_demographics,
    get_recent_observations,
)

# ── Model selection ────────────────────────────────────────────────────────────
# Set HEALTHCARE_AGENT_MODEL in your .env to switch models.
#
# All models are handled via LiteLLM. Use the appropriate prefix:
#   HEALTHCARE_AGENT_MODEL=gemini/gemini-2.5-flash   (Google AI Studio, default)
#   HEALTHCARE_AGENT_MODEL=openai/gpt-4o
#   HEALTHCARE_AGENT_MODEL=anthropic/claude-sonnet-4-6
#   HEALTHCARE_AGENT_MODEL=vertex_ai/gemini-2.5-flash
# ──────────────────────────────────────────────────────────────────────────────
_model_name = os.getenv("HEALTHCARE_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

root_agent = Agent(
    name="clinical_copilot_agent",
    model=_model,
    description=(
        "A clinical assistant that queries a patient's FHIR health record "
        "to answer questions about demographics, medications, conditions, and observations."
    ),
    instruction=("""You are a healthcare orchestration assistant.

    Delegate tasks to specialist agents:
    - triage_agent
    - diagnosis_agent
    - medication_agent
    - lab_analysis_agent
    - insurance_agent
    - care_plan_agent

    Combine outputs into a unified clinical response."""),
    tools=[
        get_patient_demographics,
        get_active_medications,
        get_active_conditions,
        get_recent_observations,
        AgentTool(agent=triage_agent),
        AgentTool(agent=diagnosis_agent),
        AgentTool(agent=medication_agent),
        AgentTool(agent=lab_analysis_agent),
        AgentTool(agent=insurance_agent),
        AgentTool(agent=care_plan_agent),
    ],
    # Runs before every LLM call.
    # Reads fhir_url, fhir_token, and patient_id from A2A message metadata
    # and writes them into session state so tools can call the FHIR server.
    before_model_callback=extract_fhir_context,
)
