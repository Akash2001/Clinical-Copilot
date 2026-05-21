import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from shared.fhir_hook import extract_fhir_context
from shared.tools import (
    get_active_conditions,
    get_active_medications,
    get_patient_demographics,
    get_recent_observations,
)
_model_name = os.getenv("HEALTHCARE_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

insurance_agent = Agent(
    name="insurance_agent",
    model=_model,
    instruction="""
    Validate insurance eligibility,
    coverage, claims, and prior authorization requirements.
    """,
    before_model_callback=extract_fhir_context,
    tools=[
        # verify_coverage_tool,
        # claims_lookup_tool,
    ],
)
