# utils/ui_components.py

"""
Contains reusable Streamlit components to build the user interface.
This keeps the main `app.py` file clean and organized, promoting a modular
frontend architecture.
"""

import streamlit as st
import json

def display_chat_message(role: str, content: str, avatar: str):
    """
    Displays a single chat message in the Streamlit interface.

    Args:
        role (str): The role of the message sender ('user' or 'assistant').
        content (str): The message content.
        avatar (str): An emoji or URL for the avatar.
    """
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

def display_json_decision(decision_data: dict):
    """
    Renders the structured JSON decision in a formatted way in the sidebar
    or an expander.

    Args:
        decision_data (dict): The JSON output from the LLM.
    """
    if not decision_data:
        return

    st.sidebar.subheader("📝 Claim Decision")

    decision = decision_data.get("decision", "N/A")
    color = "green" if decision == "Approved" else "red" if decision == "Rejected" else "orange"
    
    st.sidebar.markdown(f"**Status:** <span style='color:{color}; font-weight:bold;'>{decision}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Amount:** {decision_data.get('amount', 'N/A')}")
    st.sidebar.markdown(f"**Confidence:** {decision_data.get('confidence_score', 'N/A') * 100:.2f}%")

    with st.sidebar.expander("Justification & Sources", expanded=True):
        justifications = decision_data.get("justification", [])
        if justifications:
            for just in justifications:
                st.info(f"**Clause:** \"{just.get('text', '...')}\"\n\n**Source:** `{just.get('source_file', 'N/A')}` (Page: {just.get('page_number', 'N/A')})")
        else:
            st.write("No specific clauses cited.")

    with st.sidebar.expander("Reasoning Steps"):
        reasoning = decision_data.get("reasoning_steps", [])
        if reasoning:
            for i, step in enumerate(reasoning):
                st.markdown(f"{i+1}. {step}")
        else:
            st.write("No reasoning steps provided.")

    with st.sidebar.expander("Suggested Follow-ups"):
        follow_ups = decision_data.get("follow_up_questions", [])
        if follow_ups:
            for q in follow_ups:
                st.markdown(f"- *{q}*")
        else:
            st.write("No follow-up questions suggested.")

def display_conversation_history(history: list):
    """
    Displays the entire chat history in the main panel.

    Args:
        history (list): A list of chat messages from st.session_state.
    """
    for message in history:
        role = message["role"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            # If the assistant message contains a JSON decision, display it
            if role == "assistant" and "json_decision" in message:
                with st.expander("View Structured Decision"):
                    st.json(message["json_decision"])

def get_file_uploader():
    """
    Creates and returns the Streamlit file uploader component.
    """
    return st.sidebar.file_uploader(
        "Upload Insurance Documents",
        type=["pdf", "docx", "pptx", "txt", "eml", "jpg", "png"],
        accept_multiple_files=True,
        help="Upload policy documents, medical reports, and any relevant files."
    )
