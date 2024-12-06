from dotenv import load_dotenv

_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool 
import gradio as gr

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Define tools for the flow
@tool
def ask_email():
    """
    This tool prompts the user to provide their email address.
    """
    return "Please provide your email address."

@tool
def check_email_format(email):
    """
    This tool verifies if the provided email address is valid and exists in database or not.
    Args:
        email (str): The email address provided by the user.
    Returns:
        str or None: The email address if valid, otherwise None.
    """
    if not email or "@" not in email:
        return 'email is incorrect'  
    return 'email is correct' 

@tool
def send_code(email):
    """
    This tool sends a verification code to the verified provided email address.
    Args:
        email (str): The email address to which the code is sent.
    Returns:
        str: A message indicating that the code has been sent.
    """
    return f"Verification code has been sent to {email}. Please enter the code."

@tool
def verify_email(code):
    """
    This tool verifies if the code provided by the user is correct and same as the one sent to email address or not.
    Args:
        code (str): The verification code provided by the user.
    Returns:
        str: A message indicating whether the code is correct or incorrect.
    """
    if str(code) == '1234':  # Assume 1234 is the correct code for testing purposes
        return "code is same with the one sent to email"
    return "code is incorrect"
    

@tool
def ask_password(args=None):
    """
    This tool prompts the user to enter their new password for first time.
    """
    return "Please type your new password for first time."

@tool
def retype_password(args=None):
    """
    This tool prompts the user to retype password provided earlier for second time for confirmation.
    It has to be called immediately after ask_password().
    """
    return "Please retype your password for confirmation."

@tool
def conform_passwords(state: AgentState):
    """
    This tool confirms that both passwords requested from user are identical.
    It uses the last two human messages from the agent state.
    Args:
        state (AgentState): The current state of the agent containing messages.
    Returns:
        str: A message indicating whether the passwords match or not.
    """
    human_messages = [msg.content for msg in state['messages'] if isinstance(msg, HumanMessage)][-2:]
    if len(human_messages) == 2 and human_messages[0] == human_messages[1]:
        return "Passwords match."
    return "Passwords do not match. Please try again."

@tool
def record_password():
    """
    This tool records the new password.

    Returns:
        str: A message indicating whether the password change was successful.
    """
    return "Password has been changed successfully."
    

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        
        # Graph nodes
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        
        # Conditional edges and flow control
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        self.graph = graph.compile(checkpointer=checkpointer)  # Use the checkpointer passed as a parameter
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t['name']
            tool_args = t['args']
            print(f"Calling: {tool_name} with args: {tool_args}")
            result = self.tools[tool_name](*tool_args if isinstance(tool_args, list) else [tool_args])
            results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
        print("Back to the model!")
        return {'messages': results}

def chat_interface(user_input, history):
    if not history:
        history = []
    messages = []
    for h in history:
        messages.append(HumanMessage(content=h[0]))
        messages.append(AIMessage(content=h[1]))
    
    messages = messages + [HumanMessage(content=user_input)]
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, tools, checkpointer=memory, system=prompt)
        result = abot.graph.invoke({"messages": messages}, {"configurable": {"thread_id": "1"}})
        ai_response = result['messages'][-1].content
    history.append((user_input, ai_response))
    return history, history

if __name__ == "__main__":

    prompt = """
                You are a smart assistant tasked with guiding the user through changing their password, if they want. 
                    You have the following tools at your disposal:
                        1. ask_email: Prompt the user to provide their email address.
                        2. check_email_format: Verify if the provided email address format is valid.
                        3. send_code: Send a verification code to the format-verified email address.
                        4. verify_email: check the code provided by the user is same as sent code to email.
                        5. ask_password: Ask the user to provide a new password for first time.
                        6. retype_password: Ask the user to retype new password for second time for confirmation.
                        7. conform_passwords: Check if the two passwords match.
                        8. record_password: Record the new password and finalize the process.

                    the order of these tools is important. 
                    When appropriate, call these tools to proceed through the steps.
                    Guide the user step-by-step until the password change process is completed.
                    if user provides wrong answer ask 2 times again. if he can not provide right answer, return to the first step which is asking for email address.
                    """

    model = ChatOpenAI(model="gpt-4o-2024-08-06")
    tools = [ask_email, check_email_format, send_code, verify_email, ask_password, retype_password, conform_passwords, record_password]

    gr.Interface(fn=chat_interface, inputs=["text", "state"], outputs=["chatbot", "state"], live=False).launch(server_name="0.0.0.0", server_port=8914)
