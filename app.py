# Contribution for Mini Project 2 - Part 3: Cypress Zhang, xiaopz2@uw.edu

# Import the necessary libraries
import streamlit as st
from openai import OpenAI
from agents import Head_Agent

st.title("Mini Project 2: Streamlit Chatbot")


head_agent = Head_Agent(openai_key=st.secrets["openai_key"], pinecone_key=st.secrets["pinecone_key"], pinecone_index_name="rag")


# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    # return: A formatted string representation of the conversation.
    # ... (code for getting conversation history)
    conversation_history = ""
    for message in st.session_state.messages:
        conversation_history += f"{message['role']}: {message['content']}\n"
    return conversation_history

# Check for existing session state variables
if "openai_model" not in st.session_state:
    # ... (initialize model)
    st.session_state.openai_model = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    # ... (initialize messages)
    st.session_state.messages = []

# Display existing chat messages
# ... (code for displaying messages)
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.chat_message("user").markdown(message['content'])
    else:
        st.chat_message("assistant").markdown(message['content'])

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # ... (append user message to messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ... (display user message)
    st.chat_message("user").markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        # ... (send request to OpenAI API)
        conversation = get_conversation()

        ai_response = head_agent.main_loop(prompt, conversation, mode='serious')
        # ... (get AI response and display it)
        st.markdown(ai_response)

    # ... (append AI response to messages)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
