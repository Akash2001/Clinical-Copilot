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

diagnosis_agent = Agent(
    name="diagnosis_agent",
    model=_model,
    instruction="""
    Analyze symptoms, vitals, labs, and conditions.
    Suggest possible diagnoses with confidence levels.
    Do not provide definitive diagnosis.
    """,
    before_model_callback=extract_fhir_context,
    tools=[
        get_active_conditions,
        get_recent_observations,
        get_active_medications,
    ]
)