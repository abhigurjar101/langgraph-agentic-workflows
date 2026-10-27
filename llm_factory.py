"""Explicit offline default and optional NVIDIA-hosted model client."""
import os

def get_chat_llm(model=None, temperature=0.7, timeout=20):
    if os.getenv("AGENT_MODE", "offline").lower() == "offline":
        return None
    if os.getenv("AGENT_MODE", "offline").lower() != "live":
        raise ValueError("AGENT_MODE must be offline or live")
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    model = model or os.getenv("NVIDIA_MODEL", "").strip()
    if not api_key or not model:
        raise ValueError("Live mode requires NVIDIA_API_KEY and NVIDIA_MODEL")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, api_key=api_key,
                      base_url="https://integrate.api.nvidia.com/v1",
                      temperature=temperature, timeout=timeout, max_retries=0)

def get_writer_llm(temperature=0.7):
    return get_chat_llm(temperature=temperature)

def get_reviewer_llm(temperature=0.2):
    return get_chat_llm(temperature=temperature)
