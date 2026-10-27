"""Minimal UI for the existing LangGraph workflow."""
import os
import uuid
import streamlit as st
from demo import initial_state
from unified_agent import build_master_agent

st.set_page_config(page_title="Agentic workflow lab", page_icon="🔀")
st.title("Agentic workflow lab")
st.caption("Educational example with synthetic documents. No official institution affiliation.")
st.info("Mode: " + os.getenv("AGENT_MODE", "offline") + ". Offline output is extractive or simulated. Live mode uses a remote model.")
if "agent" not in st.session_state:
    st.session_state.agent = build_master_agent()
    st.session_state.thread_id = str(uuid.uuid4())
question = st.text_input("Ask about demo attendance, tuition, or request a content draft")
if st.button("Run workflow") and question.strip():
    try:
        result = st.session_state.agent.invoke(initial_state(question),
                    config={"configurable": {"thread_id": st.session_state.thread_id}})
        st.write(result.get("final_output") or "Workflow ended without an approved draft.")
        st.json({key: result.get(key) for key in ["query_type", "safety_scores", "attempt", "is_approved"]})
    except Exception:
        st.error("Workflow failed. Check local logs and model configuration.")
