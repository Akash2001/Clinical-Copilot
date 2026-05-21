import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from shared.fhir_hook import extract_fhir_context
from shared.tools import (
    get_active_conditions,
    get_active_medications,
)
_model_name = os.getenv("HEALTHCARE_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

medication_agent = Agent(
    name="medication_agent",
    model=_model,
    instruction="""
    Review medications for:
    - contraindications
    - interactions
    - duplicate therapies
    - allergy risks
    """,
    before_model_callback=extract_fhir_context,
    tools=[
        get_active_medications,
        get_active_conditions,
    ]
)