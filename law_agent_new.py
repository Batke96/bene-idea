import streamlit as st
import os
import uuid
import yaml
import requests
from yaml.loader import SafeLoader

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dataclasses import dataclass, field
from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from typing import Literal, List, Dict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


def gemma2(messages: List[dict], model: str = "gemma2:2b", stream: bool = False) -> str:
    """
    Sends messages to the local Gemma model and returns the text response.
    """
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
        "stream": stream
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get('message', {}).get('content', "")
    else:
        return "Es gab einen Fehler bei der Anfrage an das Gemma-Modell."

def call_gemma2(messages: List[SystemMessage]) -> AIMessage:
    """
    Converts a list of SystemMessage, HumanMessage, AIMessage instances to the
    Gemma role/content format, calls gemma2, and returns an AIMessage response.
    """
    gemma_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            # Default fallback
            role = "user"
        gemma_messages.append({"role": role, "content": msg.content})

    response_text = gemma2(gemma_messages)
    return AIMessage(content=response_text)

# -----------------------------
# 2) Hardcoded Prompts
# -----------------------------

HARDCODED_SYSTEM_PROMPT = """\
Du bist ein hilfsbereiter und prägnanter Assistent eines Anwalts, der ein möglichst realistisches Interview mit einem potenziellen Mandanten durchführt. 
Das Gespräch sollte sich für den Nutzer möglichst natürlich und flüssig anfühlen, als würdet ihr euch unterhalten. 
Bitte stelle deine Fragen nacheinander, sodass die Unterhaltung Schritt für Schritt verläuft.

Das Gespräch ist auf 5 Fragen begrenzt. Dein Ziel ist es, die folgenden Punkte herauszufinden:
1. Was ist die potentielle Schadenssumme (z.B. Gehaltsinformation bei Kündigung)?
2. Verfügt der Kunde über eine Rechtsschutzversicherung?
3. Wie viele Personen sind auf beiden Seiten beteiligt?

Beschränke die Konversation auf diese Themen, aber formuliere deine Fragen so, dass es wie ein realistisches Interview klingt. 
Bitte antworte immer auf Deutsch. 

Sobald du alle benötigten Informationen hast oder deine 5 Fragen aufgebraucht sind, 
beende das Gespräch mit dem Satz:
"Thank you for completing the interview. Your responses have been recorded."
"""

HARDCODED_SUMMARY_PROMPT = """\
Bitte erstellen Sie eine Zusammenfassung des obigen Gesprächs im Kontext der ersten Interaktion eines Anwalts. 
Berücksichtigen Sie insbesondere folgende Punkte:
1. Potentielle Schadenssumme
2. Rechtsschutzversicherung
3. Wie viele Personen sind auf beiden Seiten beteiligt?

Die Konversation ist auf maximal 5 Fragen beschränkt. 
Stellen Sie sicher, dass alles klar und deutlich in Deutsch formuliert ist, 
damit ein menschlicher Sachbearbeiter die Informationen schnell erfassen kann.
"""

# -----------------------------
# 3) Initialize Streamlit Session State
# -----------------------------

if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = HARDCODED_SYSTEM_PROMPT
if 'output_prompt' not in st.session_state:
    st.session_state.output_prompt = HARDCODED_SUMMARY_PROMPT
if 'conversation' not in st.session_state:
    st.session_state.conversation = []  # List of HumanMessage and AIMessage
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'question_count' not in st.session_state:
    st.session_state.question_count = 0

# -----------------------------
# 4) Helper Functions
# -----------------------------

def generate_summary(conversation: List, output_prompt: str) -> str:
    """
    Generates a summary of the conversation using the Gemma model based on output_prompt.
    """
    # Filter out SystemMessages
    filtered_conversation = [msg for msg in conversation if not isinstance(msg, SystemMessage)]

    prompt_messages = [
        SystemMessage(content="You are a helpful assistant that summarizes conversations."),
    ] + filtered_conversation + [
        HumanMessage(content=output_prompt)
    ]

    try:
        response = call_gemma2(prompt_messages)  # Use our Gemma call wrapper
        return response.content
    except Exception as e:
        return f"Error generating summary: {e}"

def conclude_interview() -> AIMessage:
    """
    Generates a conclusion message for the interview.
    """
    conclusion = "Thank you for completing the interview. Your responses have been recorded."
    return AIMessage(content=conclusion)

def reset_conversation():
    """
    Resets the conversation and related states.
    """
    st.session_state.conversation = []
    st.session_state.summary = ""
    st.session_state.interview_complete = False
    st.session_state.initialized = False
    st.rerun()

def initiate_first_question():
    """
    Initiates the interview by sending the system prompt to Gemma and generating the first question.
    """
    if st.session_state.system_prompt and not st.session_state.initialized:
        try:
            initial_prompt = [SystemMessage(content=st.session_state.system_prompt)]
            ai_response = call_gemma2(initial_prompt)  # Call Gemma
            if isinstance(ai_response, AIMessage):
                st.session_state.conversation.append(ai_response)
                st.session_state.initialized = True
        except Exception as e:
            error_message = AIMessage(content=f"Error initiating interview: {e}")
            st.session_state.conversation.append(error_message)
            st.session_state.initialized = True

def end_interview():
    """
    Marks the interview as completed in session state and triggers summary display.
    """
    st.session_state.interview_complete = True

# -----------------------------
# 5) Main Chat UI
# -----------------------------

def main():
    st.set_page_config(layout="centered")
    st.title("Law Assistant Interview")

    # Initiate the first question if not already done
    if not st.session_state.initialized:
        initiate_first_question()

    # If the interview is complete, show final summary and stop
    if st.session_state.interview_complete:
        st.header("Interview Complete")
        st.write(st.session_state.summary)
        if st.button("Reset Conversation"):
            reset_conversation()
        return

    # Display conversation so far
    for msg in st.session_state.conversation:
        if isinstance(msg, HumanMessage):
            st.chat_message("human").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("ai").write(msg.content)

    st.button("Interview abschließen", on_click=end_interview)

    # Chat input
    prompt = st.chat_input("Your Answer:")
    if prompt:
        # User's message
        user_message = HumanMessage(content=prompt)
        st.session_state.conversation.append(user_message)
        st.chat_message("human").write(prompt)

        # Generate updated summary in the background
        st.session_state.summary = generate_summary(
            st.session_state.conversation, 
            st.session_state.output_prompt
        )

        # Get AI's next response
        with st.spinner('AI is thinking...'):
            try:
                prompt_messages = [SystemMessage(content=st.session_state.system_prompt)] + st.session_state.conversation
                ai_response = call_gemma2(prompt_messages)

                if isinstance(ai_response, AIMessage):
                    st.session_state.conversation.append(ai_response)
                    st.chat_message("ai").write(ai_response.content)

                    # Naive approach: check if AI's response contains a question mark.
                    # If yes, increment question_count. Adjust logic to your needs.
                    if "?" in ai_response.content:
                        st.session_state.question_count += 1

                    # If the AI has asked 5 questions, end the interview automatically
                    if st.session_state.question_count >= 10:
                        # Update summary one last time
                        st.session_state.summary = generate_summary(
                            st.session_state.conversation,
                            st.session_state.output_prompt
                        )
                        end_interview()

                else:
                    error_message = AIMessage(content="Ich kann derzeit keine Antwort generieren.")
                    st.session_state.conversation.append(error_message)
                    st.chat_message("ai").write(error_message.content)
            except Exception as e:
                error_message = AIMessage(content=f"Error generating response: {e}")
                st.session_state.conversation.append(error_message)
                st.chat_message("ai").write(error_message.content)

    st.button("Reset Conversation", on_click=reset_conversation)

if __name__ == "__main__":
    main()
