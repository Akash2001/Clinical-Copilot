import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from shared.fhir_hook import extract_fhir_context
from shared.tools import (
    get_active_conditions,
    get_active_medications,
    get_recent_observations,
)
_model_name = os.getenv("HEALTHCARE_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

care_plan_agent = Agent(
    name="care_plan_agent",
    model=_model,
    instruction="""
    Generate follow-up care plans,
    lifestyle recommendations,
    medication adherence guidance,
    and next clinical steps.
    """,
    before_model_callback=extract_fhir_context,
    tools=[
        get_active_conditions,
        get_active_medications,
        get_recent_observations,
    ]
)