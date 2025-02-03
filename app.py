import streamlit as st
import time
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.title("🧠 BrainAI")
st.caption("🚀 Your own AI Neurologist with SuperPowers!!")

# Common user query suggestions
suggestions = [
    "What are the early symptoms of a brain tumor?",
    "How is a brain tumor diagnosed?",
    "What are the treatment options for brain tumors?",
    "Can a brain tumor be non-cancerous?",
    "What lifestyle changes can help manage brain tumors?"
]

# Display suggestions in rows and keep them fixed at the top
# st.write("### 💡 Common Questions")
suggestion_container = st.container()
with suggestion_container:
    for query in suggestions:
        if st.button(query, key=query):
            st.session_state["user_input"] = query
            st.rerun()

# Initiate chat engine
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt
system_prompt = SystemMessagePromptTemplate.from_template("""
    You are BrainAI, an AI-powered neurologist assistant designed to provide non-emergency guidance, education, 
    and support for neurological health. Your expertise includes brain anatomy, neurological disorders (e.g., 
    epilepsy, Alzheimer’s, brain tumors, migraines), symptoms, diagnostics, and general brain health tips. 
    Always prioritize ethical guidelines, clarify your limitations, and emphasize consulting a licensed professional 
    for personal care. Answer only in English language.
""")

# Session management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "assistant", "content": "Hello! How can I assist you with brain health today?"}]

# Chat container
chat_container = st.container()

# Display messages with animation
def display_text_with_animation(text):
    message_placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        message_placeholder.markdown(displayed_text)
        time.sleep(0.01)

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            if "<think>" in message["content"]:
                parts = message["content"].split("</think>")
                think_content = parts[0].replace("<think>", "").strip()
                actual_response = parts[-1].strip()
                
                with st.expander("🔍 View AI's Thinking Process"):
                    st.markdown(f"*Internal Analysis:*\n{think_content}")
                
                display_text_with_animation(actual_response)
            else:
                display_text_with_animation(message["content"])

# Chat input
user_query = st.chat_input(" Message . . . ")

# If a suggestion was selected, use it as the input
if "user_input" in st.session_state:
    user_query = st.session_state["user_input"]
    del st.session_state["user_input"]

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "assistant":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})

    with st.spinner("🧠 Thinking ..."):
        prompt_chain = build_prompt_chain()
        raw_response = generate_ai_response(prompt_chain)

        st.session_state.message_log.append({
            "role": "assistant",
            "content": raw_response
        })

    st.rerun()
