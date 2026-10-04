"""Groq + CrewAI configuration (single place to change the model)."""
import os

# Must be set BEFORE crewai is imported anywhere (avoids telemetry network calls).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

GROQ_MODEL_ID = "openai/gpt-oss-120b"          # the model id exactly as Groq names it
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


def get_api_key() -> str | None:
    """Streamlit secrets first, then environment variable."""
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def build_llm(api_key: str | None = None):
    """Return a CrewAI LLM that talks to Groq.

    CrewAI 1.x no longer bundles LiteLLM, and has no native 'groq' provider.
    Groq exposes an OpenAI-compatible API, so we use CrewAI's native OpenAI
    provider with Groq's base_url. CrewAI strips the first 'openai/' prefix,
    so we write it twice to keep Groq's real id 'openai/gpt-oss-120b'.
    """
    from crewai import LLM

    key = api_key or get_api_key()
    if not key:
        raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit secrets.")
    return LLM(
        model=f"openai/{GROQ_MODEL_ID}",
        base_url=GROQ_BASE_URL,
        api_key=key,
        temperature=0.2,
        max_tokens=2500,
        timeout=90,
        max_retries=3,
    )
