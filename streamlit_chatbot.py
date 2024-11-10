import streamlit as st
from openai import OpenAI
import os

# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found in environment variables. Please set 'OPENAI_API_KEY'.")
    st.stop()

# Initialize chatbot instance
chatbot = OpenAI(api_key=api_key)

# Initialize system message for the chatbot
system_msg = ""

# Check if the conversation exists in session_state, if not initialize it
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "system", "content": system_msg}]

def get_chatbot_response(user_input):
    # Append user input to conversation
    user_message = {"role": "user", "content": user_input}
    st.session_state.conversation.append(user_message)

    # Call OpenAI API to generate response
    response = chatbot.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=st.session_state.conversation
    )

    # Append chatbot response to conversation
    bot_message = {"role": "assistant", "content": response.choices[0].message.content}
    st.session_state.conversation.append(bot_message)
    return bot_message["content"]

# Streamlit UI
st.title("AI Chatbot")

# Create a form to submit user input and avoid the error
with st.form(key="chat_form"):
    user_input = st.text_input("You:", key="user_input")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # Get chatbot response and update the conversation
    chatbot_response = get_chatbot_response(user_input)

# Display conversation history
st.write("### Conversation")
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"Chatbot: {message['content']}")
