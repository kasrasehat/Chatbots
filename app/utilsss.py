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
from langchain_core.tools import tool 
import requests
import json


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