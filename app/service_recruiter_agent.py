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
import requests
import json
import fastapi
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import traceback
import json
import re
# from utilsss import Agent, AgentState, search_candidate
from typing import Dict, Any, List
from langchain_core.tools import tool 


# Initialize FastAPI app
app = FastAPI(
    title="Recruiter Agent Service",
    description="An API for interacting with recruiter to find apropriate candidates.",
    version="1.0.0"
)

# 🔹 Detect and Serialize Message Type (Ensures No Data Loss)
def serialize_message(message: Any) -> Dict[str, Any]:
    base_data = {
        "type": message.__class__.__name__,
        "content": message.content,
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
        "response_metadata": getattr(message, "response_metadata", {}),
    }

    # Handle AIMessage-specific fields
    if isinstance(message, AIMessage):
        base_data.update({
            "id": getattr(message, "id", None),
            "usage_metadata": getattr(message, "usage_metadata", {}),
            "tool_calls": getattr(message, "tool_calls", []),
        })

    # Handle ToolMessage-specific fields
    elif isinstance(message, ToolMessage):
        base_data.update({
            "tool_call_id": message.tool_call_id
        })

    return base_data

def validate_and_fix_usage_metadata(usage_metadata):
    """ Ensure `usage_metadata` contains required fields with default values. """
    if not isinstance(usage_metadata, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    return {
        "input_tokens": usage_metadata.get("input_tokens", 0),
        "output_tokens": usage_metadata.get("output_tokens", 0),
        "total_tokens": usage_metadata.get("total_tokens", 0),
    }

# 🔹 Detect and Deserialize Message Type (Ensures No Data Loss)
def deserialize_message(message_json: Dict[str, Any]) -> Any:
    message_type = message_json["type"]

    # Deserialize AIMessage
    if message_type == "AIMessage":
        return AIMessage(
        content=message_json["content"],
        additional_kwargs=message_json.get("additional_kwargs", {}),
        response_metadata=message_json.get("response_metadata", {}),
        id=message_json.get("id", ""),
        tool_calls=message_json.get("tool_calls", []),
        usage_metadata=validate_and_fix_usage_metadata(message_json.get("usage_metadata", {}))
    )

    # Deserialize HumanMessage
    elif message_type == "HumanMessage":
        return HumanMessage(
            content=message_json["content"],
            additional_kwargs=message_json.get("additional_kwargs", {}),
            response_metadata=message_json.get("response_metadata", {})
        )

    # Deserialize ToolMessage
    elif message_type == "ToolMessage":
        return ToolMessage(
            content=message_json["content"],
            tool_call_id=message_json.get("tool_call_id", "")
        )

    else:
        raise ValueError(f"Unknown message type: {message_type}")

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
    url = "https://dev-hiring-candidate.berryonmars.com/Admin/candidate/GetAnonimousByCustomFieldList?skip=0&take=5"

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
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return response.text  # Return parsed JSON response
    except requests.exceptions.RequestException as e:
        return f"Error occurred: {str(e)}"

    
class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        # self.user_input = user_input
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
                            You are an AI assistant responsible for retrieving candidate profiles and returning them in a **strictly structured JSON format**. Your response **must always conform exactly** to the required format.

                            ---

                            ### **Candidate Data Input Format**
                            When receiving candidate data in the following format:
                            ```json
                            {
                                "isSuccess": true,
                                "value": {
                                    "candidates": [],
                                    "hasNextPage": false
                                },
                                "error": null
                            }
                            
                            If candidates exist, return:
                            {
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
                            
                            If no candidates exist, return:
                            {
                                "response": "There are no candidates with this criteria within our database. {{CANDIDATE_LIST}}",
                                "data": {
                                    "CANDIDATE_LIST": []
                                }
                            }
                            note: return only the content json. do not add any additional text including back ticks or json word.
                            '''
            messages = [SystemMessage(content=system_message)] + messages
            message = self.model.invoke(messages)
            
        return {'messages': [message]}    

def get_response(state):
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
                       api_key="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")
    tools = [search_candidate]


    # messages = conversation_history + [HumanMessage(content=user_input)]
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, tools, checkpointer=memory, system=prompt)
        result = abot.graph.invoke({"messages": conversation_history}, {"configurable": {"thread_id": "1"}})
        agent_state = result
        
    return agent_state  


def fix_json_quotes(input_string):
    """
    Fix improperly escaped double quotes and ensure the JSON string is valid.
    """
    # Regex to fix improperly escaped double quotes inside content values
    pattern = r'\\"'
    fixed_string = re.sub(pattern, '"', input_string)

    # Regex to fix mismatched quotes like "d and "re
    fixed_string = re.sub(r'"d', "'d", fixed_string)
    fixed_string = re.sub(r'"re', "'re", fixed_string)

    return fixed_string

# Route to redirect root to API documentation
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

# Main chatbot endpoint
@app.post("/recruiter")
async def recruiting_endpoint(
    agent_state: str = Form(...)  # Chat history as a string
):
    """
    - `agent_state`: String containing the chat history (JSON format).
    """
    try:
        # Convert the agent_state string to a list
        try:
            # Fix JSON using regex
            # Replace single quotes with double quotes to make it valid JSON
            stringified_data = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', agent_state)
            agent_state = json.loads(stringified_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for chat history.")

       # Initialize an instance of AgentState
        agent_state: AgentState = agent_state
       # Deserialize messages, ensuring tool_calls are properly structured
        agent_state_dict = {
            "messages": [deserialize_message(msg) for msg in agent_state['messages']],
            "flow_state": agent_state['flow_state']
        }
        # Send request to agent
        updated_agent_state = get_response(agent_state_dict)
        # Convert messages to JSON
        final_agent_state = {
            "messages": [serialize_message(msg) for msg in updated_agent_state["messages"]],
            "flow_state": updated_agent_state.get("flow_state", {})
        }
        
        # # Convert to Python dict
        # try:
        #     json_agent_state = json.loads(final_agent_state)
        #     print("✅ Successfully loaded JSON:", json_agent_state)
        # except json.JSONDecodeError as e:
        #     print("❌ JSON Error:", str(e))

        return JSONResponse(content=final_agent_state)

    except Exception as e:
        # Handle errors gracefully
        full_traceback = traceback.format_exc()
        return JSONResponse(
            content={"error": str(e), "traceback": full_traceback},
            status_code=500
        )

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8080)
    