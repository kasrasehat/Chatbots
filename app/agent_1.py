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
from langgraph.graph import StateGraph, END, START
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


class Email_Agent:
    def __init__(self, model, tools, checkpointer, system="", user_input='', router_prompt=""):
        self.system = system
        self.user_input = user_input
        self.router_prompt = router_prompt
        # self.model_rag = model
        
        graph = StateGraph(AgentState)
        
        # Graph nodes
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_node("llm1", self.call_openai1)
        # Conditional edges and flow control
        graph.add_conditional_edges(START, self.router, {'recruiter': "llm", 'job_post_maker': END, 'meeting_scheduler': END, 'general_questions':END})
        graph.add_conditional_edges("llm", self.exists_action, {'take_action': "action", False: END})
        graph.add_edge("action", "llm1")
        graph.add_edge("llm1", END)
        
        self.graph = graph.compile(checkpointer=checkpointer)  # Use the checkpointer passed as a parameter
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.router_model = model

    def router(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.router_prompt)] + messages
            message = self.router_model.invoke(messages)
        return message.content

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