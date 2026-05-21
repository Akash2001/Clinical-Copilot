import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from shared.fhir_hook import extract_fhir_context
from shared.tools import (
    get_active_conditions,
    get_recent_observations,
)

_model_name = os.getenv("HEALTHCARE_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

triage_agent = Agent(
    name="triage_agent",
    model=_model,
    instruction="""
    You summarize patient symptoms and determine urgency level:
    - emergency
    - urgent
    - routine
    """,
    before_model_callback=extract_fhir_context,
    tools=[
        get_recent_observations,
        get_active_conditions,
    ]
)