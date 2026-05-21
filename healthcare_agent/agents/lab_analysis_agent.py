import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from shared.fhir_hook import extract_fhir_context
from shared.tools import (
    get_recent_observations,
)
_model_name = os.getenv("HEALTHCARE_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

lab_analysis_agent = Agent(
    name="lab_analysis_agent",
    model=_model,
    instruction="""
    Interpret lab values and observations.
    Highlight abnormal trends and explain clinical meaning.
    """,
    before_model_callback=extract_fhir_context,
    tools=[
        get_recent_observations,
    ]
)