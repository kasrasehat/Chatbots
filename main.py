from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("error: API key not found in environment variables")

chatbot = OpenAI(api_key=api_key)
# Initialize the system message for the OpenAI API
system_msg = ""
# Set up the message to send to OpenAI
messages = [
    {"role": "system", "content": system_msg},
]
# {"role": "user", "content": text}
conversation = [messages[0]]


def get_chatbot_response(user_input):
    message = {
               "role": "user",
               "content": user_input
               }
    conversation.append(message)
    response = chatbot.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=conversation
    )
    conversation.append(response.choices[0].message.content)
    return response.choices[0].message.content


def chat():
    # Call OpenAI API to generate structured resume information
    while True:
        user_input = input("you: ")
        if user_input == "exit":
            print("Chatbot: Goodbye!")
            break
        response = get_chatbot_response(user_input)


if __name__ == "__main__":
    chat()




