import numpy as np
from tavily import TavilyClient
import os
import openai
from openai import OpenAI
import gradio as gr
import urllib.parse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib
import sys
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing
import time
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Required
import operator
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool 
import requests
import json
import traceback
import logging

# Configure logging
logging.basicConfig( 
    level=logging.INFO,  # Log INFO and higher severity levels
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    flow_state: Required[str]  # Track which stage the agent is in


# Define tools for the flow
@tool
def search_candidate(criteria=None):
    """
    Searches for candidates based on the provided criteria using an API request.

    This function sends a POST request to the hiring candidate API endpoint, 
    querying the database for anonymous candidates that match the given criteria.

    Args:
        criteria (dict, optional): A dictionary containing search filters. 
            Example:
                {
                    "job_title": "AI engineer",
                    "city": "tehran",
                    "min_salary": "30000",
                    "max_salary": "90000"
                }

    Returns:
        dict: The API response parsed as JSON if successful.
        str: An error message if the request fails.

    Raises:
        requests.exceptions.RequestException: If the request encounters an error.

    Notes:
        - Requires a valid API authentication token in the `Authorization` header.
        - Ensure `criteria` follows the API's expected format.
    """
    url = "http://candidate-candidate:8080/admin/candidate/GetAnonimousByCustomFieldList?skip=0&take=5"

    payload = json.dumps([
        {
            "logicalOp": 0,
            "fieldValue": criteria['job_title'],
            "comparisonOp": 5,
            "fieldName": "CandidateWork.Title"
        },
        {
            "fieldName": "distance",
            "logicalOp": 0,
            "comparisonOp": 5,
            "fieldValue": criteria['city']
        },
        {
            "fieldName": "SalaryFrom",
            "logicalOp": 0,
            "fieldValue": int(criteria['min_salary']),
            "comparisonOp": 3
        },
        {
            "fieldName": "SalaryTo",
            "logicalOp": 0,
            "fieldValue": int(criteria['max_salary']),
            "comparisonOp": 4
        }
    ])

    headers = {
        'accept': '*/*',
        'Authorization': 'Bearer',  # Replace with a valid token
        'Content-Type': 'application/json'
    }

    try:
        # Log request details
        logging.info(f"Sending POST request to URL: {url}")
        logging.info(f"Headers: {json.dumps(headers, indent=2)}")
        logging.info(f"Payload: {payload}")

        response = requests.post(url, headers=headers, data=payload)

        # Log response details
        logging.info(f"Received response - Status Code: {response.status_code}")
        logging.info(f"Response Body: {response.text[:500]}")  # Log first 500 chars of response

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

        return response.text  # Return parsed JSON response

    except requests.exceptions.RequestException as e:
        error_message = f"Error occurred: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())  # Log full traceback
        return error_message

    
class Agent:
    def __init__(self, model, tools, checkpointer, system="", user_input=''):
        self.system = system
        self.user_input = user_input
        # self.model_rag = model
        
        graph = StateGraph(AgentState)
        
        # Graph nodes
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_node("llm1", self.call_openai1)
    
        # Conditional edges and flow control
        graph.set_entry_point("llm")
        graph.add_conditional_edges("llm", self.exists_action, {'take_action': "action", False: END})
        graph.add_edge("action", "llm1")
        graph.add_edge("llm1", END)
        
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
        
        if len(result.tool_calls) > 0:
            state["flow_state"] = "candidate_retriever" 
            return 'take_action'
        
        else: 
            state["flow_state"] = ""
            return False


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
   
    def call_openai1(self, state: AgentState):
        messages = state['messages']
        if self.system:
            system_message = '''
                            You are an AI assistant responsible for retrieving candidate profiles and returning them in a structured JSON format.
                            Your goal is to strictly follow this format when receive candidates in this format:
                            {"isSuccess":true,"value":{"candidates":[],"hasNextPage":false},"error":null}
                            
                            then return the candidates in this format from candidates: field from above dictionary:
                            {
                                "text": {
                                    "response": "Sure, here are our candidates: You can select two options with regards to these candidates. {{CANDIDATE_LIST}}",
                                    "data": {
                                        "CANDIDATE_LIST": [
                                            {
                                                "id": 1,
                                                "candidateWork": [{ "title": "Principal Frontend Engineer" }],
                                                "description": "Frontend engineer with 10+ years of experience with proficiency in JavaScript, TypeScript, and React.js contributed to building secure and performant applications."
                                            },
                                            {
                                                "id": 2,
                                                "candidateWork": [{ "title": "Frontend Engineer" }],
                                                "description": "Frontend engineer with 3+ years of experience with proficiency in JavaScript, TypeScript, and React.js contributed to building secure and performant applications with a focus on the best user and developer experience."
                                            }
                                        ]
                                    }
                                }
                            }
                                
                            Guidelines for Execution:

                            1- Always retrieve candidate data before responding.
                            2- Ensure the response is structured strictly in the given JSON format.
                            3- If no candidates are available, return an empty CANDIDATE_LIST, ensuring the response remains valid:    
                            {
                                "text": {
                                    "response": "There are no candidates with this criteria within our database. {{CANDIDATE_LIST}}",
                                    "data": {
                                        "CANDIDATE_LIST": []
                                    }
                                }
                            }
                            4- Do not alter the format, structure, or wording of the response key.
                            5- Ensure that each candidate entry includes:
                                    id (unique identifier)
                                    candidateWork (list of job titles)
                                    description (concise candidate summary)
                            6- If candidates exist, replace {{CANDIDATE_LIST}} with the retrieved candidate data.
                            
                            '''
            messages = [SystemMessage(content=system_message)] + messages
            message = self.model.invoke(messages)
            
        return {'messages': [message]}
    
   
def get_response(user_input, state):
    # This function now has access to the full conversation history from state
    # Construct a response based on state messages

    # Here you can interact with the LLM model using all past messages
    # Example: Generate a response using past conversation history
    conversation_history = state["messages"]

    # If using OpenAI or any LLM, you'd pass the full conversation history
    prompt = '''You are a hiring assistant responsible for gathering candidate search criteria from an employer.

                        Your goal is to ask specific questions to complete the required fields in the JSON format before retrieving candidates.

                        required JSON structure:
                        [
                            {
                                **"job_title": "AI engineer",
                                **"city": "Dusseldorf"
                                **"min_salary": 65000,
                                **"max_salary": 75000,
                            }
                        ]

                        Steps:
                        1. Ask the employer relevant questions to fill each field:
                        - "For what job are you looking for an employee?"
                        - "In which city are you looking for an employee?"
                        - "What is the minimum salary range you are looking for?"
                        - "What is the maximum salary range you are looking for?"

                        2. If the employer does not provide a clear answer:
                        - Ask **twice more** for clarification.
                        - If an answer is still unclear, set the field to `None`.

                        3. **Ensure all required fields are completed** before proceeding.

                        4. Once the JSON all keys are fully populated, **call the `search_candidate` tool** to fetch candidates based on the gathered criteria.

                        5. Do not use the search_candidate tool until all required fields are provided.

                        Your responses should be structured, clear, and assist the employer in providing accurate data. Always verify the completeness of the JSON before making any retrieval requests.
                        '''

    model = ChatOpenAI(model="gpt-4o-2024-08-06", 
                       temperature=0, 
                       api_key= os.getenv("OPENAI_API_KEY"))
    tools = [search_candidate]


    # messages = conversation_history + [HumanMessage(content=user_input)]
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, tools, checkpointer=memory, system=prompt, user_input=user_input)
        result = abot.graph.invoke({"messages": conversation_history}, {"configurable": {"thread_id": "1"}})
        agent_state = result
        
    return agent_state  


def get_demo():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>AI Chatbot to interact with Website and PDF</h1>")

        # Configuration Inputs
        with gr.Row():

            # Chat Interface
            with gr.Column():
                user_input = gr.Textbox(label="Your question:", placeholder="Type your question here...", interactive=True)
                chat_output = gr.Textbox(label="Chatbot Response", placeholder="Chatbot will respond here...", interactive=True)
                send_button = gr.Button("Send", interactive=True)

        # State to hold the agent messages
        agent_state = gr.State(value={"messages": []})

        def on_send_button(user_input, agent_state):
            # Retrieve current state of messages
            state = agent_state  # This is the current agent state holding all previous messages

            # Append the new user input as a HumanMessage to state
            state['messages'].append(HumanMessage(content=user_input))

            # Use the updated state to get a response
            agent_state = get_response(user_input, state)
            response_content = agent_state['messages'][-1].content

            # Append the model's response to the state
            response_message = AIMessage(content=response_content)
            agent_state['messages'].append(response_message)

            # Return the updated agent state and the response for the UI
            return response_content, agent_state


        send_button.click(
            fn=on_send_button,
            inputs=[user_input, agent_state],
            outputs=[chat_output, agent_state]
        )

    return demo

if __name__ == "__main__":
    demo = get_demo()
    demo.launch(server_name="127.0.0.1", server_port=8718)
    