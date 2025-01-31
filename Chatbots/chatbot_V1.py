from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("error: API key not found in environment variables")

chatbot = OpenAI(api_key="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")
# Initialize the system message for the OpenAI API
system_msg = '''you are a hiring assistant. 
             you have to ask questions from employer to complete fieldValue of these dictionaries within list due to their field name.
 example:
 [
  {
    "logicalOp": 0,
    "fieldValue": "frontend developer",
    "comparisonOp": 5,
    "fieldName": "CandidateWork.Title"
  },
  {
    "fieldName": "distance",
    "logicalOp": 0,
    "comparisonOp": 5,
    "fieldValue": "Dusseldorf"
  },
  {
    "fieldName": "SalaryFrom",
    "logicalOp": 0,
    "fieldValue": 65000,
    "comparisonOp": 3
  },
  {
    "fieldName": "SalaryTo",
    "logicalOp": 0,
    "fieldValue": 75000,
    "comparisonOp": 4
  }
]
for what job are you looking for an employee?
in which city are you looking for an employee?
What is the minimum salary range you are looking for?
What is the maximum salary range you are looking for?
if you do not get appropriate answer, ask twice. if you can not get your answer fill this field None. 
when you completed these field return json'''
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
        temperature=0,
        messages=conversation
    )
    conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content


def chat():
    # Call OpenAI API to generate structured resume information
    while True:
        user_input = input("you: ")
        if user_input in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = get_chatbot_response(user_input)
        print(f"chatbot: {response}")


if __name__ == "__main__":
    chat()




