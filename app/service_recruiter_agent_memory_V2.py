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
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Required, Optional
import operator
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, messages_from_dict, messages_to_dict
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
from typing import Dict, Any, List, TypedDict, Literal, Optional
from langchain_core.tools import tool 
import traceback
import logging
import redis
import requests
import json
import traceback
import logging
import os
from datetime import datetime, timedelta
import pytz
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
import json
from datetime import datetime
from datetime import datetime, timedelta
from typing import List, Dict
from langchain.tools import tool
from dotenv import load_dotenv
_ = load_dotenv('.env.dev')

# Configure logging
logging.basicConfig( 
    level=logging.INFO,  # Log INFO and higher severity levels
    format="%(asctime)s - %(levelname)s - %(message)s",
)

redis_client = redis.Redis(host='redis-master.tools', port=6379, db=0, decode_responses=True, password="1qaz2wsx3edc")
# , password='1qaz2wsx3edc'
# Initialize FastAPI app
app = FastAPI(
    title="Recruiter Agent Service",
    description="An API for interacting with recruiter to find apropriate candidates.",
    version="1.0.0"
)

class JobPostInput(TypedDict):
    """Fields required to create a job post."""
    title: str
    requirementsText: str
    description: str
    salaryFrom: int
    salaryTo: int
    country: str
    city: str
    jobType: Literal["fullTime", "partTime", "internship", "contract"]
    locationTypeEnum: Literal["remote", "onsite", "hybrid"]
    employerId: str

@tool
def job_creator(job_data: JobPostInput) -> Optional[str]:
    """
    Creates a job post using the employer's input and saves it into the job posting system.

    This tool is used **after all required job posting fields are collected from the employer**.
    It sends the data to an external job post API and receives a unique job code in response.

    Required fields:
    - title: Job title (e.g., "AI Engineer")
    - requirementsText: Candidate requirements in text form
    - description: Full job description
    - salaryFrom: Minimum salary
    - salaryTo: Maximum salary
    - country: Country where the job is located
    - city: City of the job
    - jobType: Type of employment (choose from: fullTime, partTime, internship, contract)
    - locationTypeEnum: Work mode (choose from: remote, onsite, hybrid)
    - employerId: Email or unique ID of the employer (must not be null)

    Returns:
    - A unique job code (string) like "67f759f4d355d64f14a8cecc" on success
    - None if the job post could not be created
    """
    url = "https://dev-hiring-employer.berryonmars.com/JobPost/CreateFromAI"

    if not job_data.get("employerId"):
        logging.error("❌ employerId is missing – job post cannot be created.")
        raise ValueError("❌ employerId is required to create a job post.")

    headers = {
        "accept": "*/*",
        "Content-Type": "application/json"
        # "Authorization": "Bearer <token>"  # Uncomment if needed
    }

    try:
        payload = json.dumps(job_data)
        logging.info(f"📤 Sending job post to {url}")
        logging.debug(f"Payload: {payload}")

        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()

        job_code = response.text.strip().strip('"')
        logging.info(f"✅ Job post created successfully — code: {job_code}")
        return job_code

    except requests.RequestException as e:
        logging.error(f"❌ Request failed while creating job post: {e}")
        return None
    except Exception as ex:
        logging.exception(f"❌ Unexpected error in job_creator: {ex}")
        return None


@tool
def calculate_free_times() -> List[Dict[str, str]]:
    """
    Retrieve available time slots within the next 7 days.

    It queries a date range starting from now and ending 7 days later.

    Parameters:
    ----------
    email : str
        The email address of the person whose calendar is being queried 
        (e.g., r.sheshbolooki@berryonmars.com).

    Returns:
    -------
    List[dict]
        A list of available free time slots with:
        {
            "date": "YYYY-MM-DD",
            "start_time": "HH:MM",
            "end_time": "HH:MM"
        }

    Example:
    --------
    >>> get_free_time_slots("r.sheshbolooki@berryonmars.com")
    [
        {"date": "2025-04-01", "start_time": "09:00", "end_time": "09:30"},
        {"date": "2025-04-01", "start_time": "09:30", "end_time": "10:00"},
        ...
    ]

    Notes:
    ------
    - The function automatically sets the query range from the current datetime 
      to 7 days ahead.
    - Assumes the API returns results in ISO 8601 format.
    - Returns an empty list on error or if no free slots are found.
    """

    # Set date range: today to 7 days later
    now = datetime.utcnow()
    start_date = now.strftime("%Y-%m-%dT%H:%M:%S")
    end_date = (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    url = f"https://dev-hiring-candidate.berryonmars.com/GetFreeEvents/r.sheshbolooki@berryonmars.com/{start_date}/{end_date}"

    headers = {
        "accept": "*/*",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        events = json.loads(response.text)
        free_slots = []

        for event in events:
            start_dt = datetime.fromisoformat(event["start"]["dateTime"])
            end_dt = datetime.fromisoformat(event["end"]["dateTime"])

            free_slots.append({
                "date": start_dt.strftime("%Y-%m-%d"),
                "start_time": start_dt.strftime("%H:%M"),
                "end_time": end_dt.strftime("%H:%M")
            })

        return free_slots

    except Exception as e:
        print(f"❌ Failed to fetch free time slots: {e}")
        return []

@tool
def schedule_meeting(email: str, reserved_time: str) -> Optional[str]:
    """
    Schedule a calendar meeting with a given attendee at a specific time.

    This tool uses the recruiting meeting scheduling system to create a calendar event.
    It receives the attendee's email and the reserved meeting time, then sends a request
    to create an event in the system.

    Parameters:
    ----------
    email : str
        The email address of the attendee who will receive the calendar invite.
    reserved_time : str
        The meeting start and end time in ISO 8601 format (e.g., "2025-04-01T14:00:00").
        The meeting is assumed to be 30 minutes long.

    Returns:
    -------
    str or None
        A success message if the meeting is scheduled successfully.
        Returns None if the operation fails.

    Example:
    --------
    >>> schedule_meeting("attendee@example.com", "2025-04-01T10:00:00")
    '✅ Meeting scheduled successfully.'

    Notes:
    ------
    - This tool assumes meetings are always 30 minutes.
    - If the reserved_time format is invalid, or the request fails, the function returns None.
    """

    # API endpoint for scheduling
    url = "https://dev-hiring-candidate.berryonmars.com/Calender/r.sheshbolooki@berryonmars.com/CreateEvent"

    # Default meeting duration: 30 minutes
    meeting_duration_minutes = 30
    try:
        from datetime import datetime, timedelta

        start_time = datetime.fromisoformat(reserved_time)
        end_time = (start_time + timedelta(minutes=meeting_duration_minutes)).isoformat()

        payload = json.dumps({
            "subject": f"Meeting with {email.split('@')[0]} from Test",
            "body": {
                "contentType": 1,
                "content": "Let's have a meeting"
            },
            "start": {
                "timeZone": "Asia/Tehran",
                "dateTime": reserved_time
            },
            "end": {
                "timeZone": "Asia/Tehran",
                "dateTime": end_time
            },
            "location": {
                "displayName": "Recruiting Meeting Scheduler"
            },
            "attendees": [
                {
                    "type": 0,
                    "emailAddress": {
                        "name": f"{email.split('@')[0]}",
                        "address": email
                    }
                }
            ],
            "allowNewTimeProposals": True
        })

        headers = {
            'accept': '*/*',
            'Content-Type': 'application/json'
            # 'Authorization': 'Bearer <token>'  # Add if needed
        }

        response = requests.post(url, data=payload, headers=headers)

        if response.status_code in [200, 201]:
            return "✅ Meeting scheduled successfully."
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Error scheduling meeting: {e}")
        return None


# 🔹 Detect and Serialize Message Type (Ensures No Data Loss)
def serialize_message(message: Any) -> Dict[str, Any]:
   return json.dumps(messages_to_dict([message]))

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
    return messages_from_dict(json.loads(message_json))

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
    # "https://dev-hiring-candidate.berryonmars.com/Admin/candidate/GetAnonimousByCustomFieldList?skip=0&take=50"

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
    def __init__(self, model, router_model, tools, checkpointer, system="", user_input='', router_prompt="", job_post_prompt="", scheduler_prompt="", general_prompt=""):
        self.router_model = router_model
        self.system = system
        self.user_input = user_input
        self.router_prompt = router_prompt
        self.job_post_prompt = job_post_prompt
        self.scheduler_prompt = scheduler_prompt
        self.general_prompt = general_prompt
        # self.model_rag = model
        
        graph = StateGraph(AgentState)
        
        # Graph nodes
        graph.add_node("job_poster", self.job_poster)
        graph.add_node("scheduler", self.call_scheduler)
        graph.add_node("general", self.general)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_node("action_scheduler", self.take_action_scheduler)
        graph.add_node("action_job_post", self.take_action_job_post)
        graph.add_node("llm1", self.call_openai1)
        # Conditional edges and flow control
        graph.add_conditional_edges(START, self.router, {'recruiter': "llm", 'job_post_maker': "job_poster", 'meeting_scheduler': "scheduler", 'general_questions':"general"})
        graph.add_conditional_edges("llm", self.exists_action, {'take_action': "action", False: END})
        graph.add_conditional_edges("job_poster", self.exists_action_job_post, {'take_action': "action_job_post", False: END})
        graph.add_conditional_edges("scheduler", self.exists_action_scheduler, {'take_action': "action_scheduler", False: END})
        graph.add_edge("action_scheduler", "scheduler")
        graph.add_edge("action_job_post", "job_poster")
        graph.add_edge("action", "llm1")
        graph.add_edge("llm1", END)
        graph.add_edge("general", END)
        
        self.graph = graph.compile(checkpointer=checkpointer)  # Use the checkpointer passed as a parameter
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.base_model = model


    def router(self, state: AgentState):
        messages = state['messages']
        
        # Main router prompt
        if self.router_prompt:
            router_messages = [SystemMessage(content=self.router_prompt)] + messages
        
        allowed_routes = ["recruiter", "job_post_maker", "meeting_scheduler", "general_questions"]

        # Invoke primary router model
        router_response = self.router_model.invoke(router_messages)
        route = router_response.content.strip().lower()

        # Check if primary model route is valid
        if route not in allowed_routes:
            print(f"[Router Warning] Primary router returned invalid route '{route}'. Invoking validator model.")

            # Limit messages to last 5 exchanges for validator
            recent_messages = messages[-5:] if len(messages) > 5 else messages[1:]
            formatted_conversation = "\n".join(
                [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_messages]
            )

            # Validator system prompt with context
            validator_prompt = f"""
            You are a validation agent tasked with ensuring correctness of routing decisions. 
            The primary router returned an invalid routing decision '{route}'.

            Carefully review the recent conversation between the user and AI assistant below, and strictly select ONE of the valid routes:

            - recruiter
            - job_post_maker
            - meeting_scheduler
            - general_questions

            Conversation:
            {formatted_conversation}

            If unsure or unclear, default immediately to 'general_questions'.

            Return ONLY ONE of the allowed routes WITHOUT ADDITIONAL TEXT.
            """

            validator_message = [SystemMessage(content=validator_prompt)]
            validator_response = self.base_model.invoke(validator_message)
            corrected_route = validator_response.content.strip().lower()

            if corrected_route in allowed_routes:
                print(f"[Validator] Corrected route to '{corrected_route}'.")
                return corrected_route
            else:
                print(f"[Validator Warning] Validator also failed with '{corrected_route}'. Defaulting to 'general_questions'.")
                return "general_questions"
    
    # If initial route is valid
        return route

    def general(self, state: AgentState):
        messages = state['messages']
        if self.general_prompt:
            messages = [SystemMessage(content=self.general_prompt)] + messages
            message = self.base_model.invoke(messages)
        return {'messages': [message], 'flow_state': 'start'}


    def job_poster(self, state: AgentState):
        messages = state['messages']
        if self.job_post_prompt:
            messages = [SystemMessage(content=self.job_post_prompt)] + messages
            message = self.model.invoke(messages)
        return {'messages': [message], 'flow_state': 'start'}
    
    def call_scheduler(self, state: AgentState):
        messages = state['messages']
        if self.scheduler_prompt:
            messages = [SystemMessage(content=self.scheduler_prompt)] + messages
            message = self.model.invoke(messages)
        return {'messages': [message], 'flow_state': 'start'}

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            message = self.model.invoke(messages)
        return {'messages': [message], 'flow_state': 'start'}

    def exists_action_scheduler(self, state: AgentState):
        result = state['messages'][-1]
        
        if len(result.tool_calls) > 0:
            state["flow_state"] = "candidate_retriever" 
            return 'take_action'
        
        else: 
            state["flow_state"] = ""
            return False


    def take_action_scheduler(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t['name']
            tool_args = t['args']
            print(f"Calling: {tool_name} with args: {tool_args}")
            result = self.tools[tool_name](*tool_args if isinstance(tool_args, list) else [tool_args])
            results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
        print("Back to the model!")
        return {'messages': results, 'flow_state': 'start'}


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
        return {'messages': results, 'flow_state': 'start'}
   
    def call_openai1(self, state: AgentState):
        messages = state['messages']
        if self.system:
            system_message = '''
                            You are an AI assistant responsible for retrieving candidate profiles and returning them in a structured JSON format.
                            Your goal is to strictly follow this format when receive candidates in this format:
                            {"isSuccess":true,"value":{"candidates":[],"hasNextPage":false},"error":null}
                            
                            then return the candidates in this format from candidates field from above dictionary:
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
            
        return {'messages': [message], 'flow_state': 'start'}
    
    def exists_action_job_post(self, state: AgentState):
        result = state['messages'][-1]
        
        if len(result.tool_calls) > 0:
            state["flow_state"] = "create_job_post" 
            return 'take_action'
        
        else: 
            state["flow_state"] = ""
            return False


    def take_action_job_post(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t['name']
            tool_args = t['args']
            print(f"Calling: {tool_name} with args: {tool_args}")
            result = self.tools[tool_name](*tool_args if isinstance(tool_args, list) else [tool_args])
            results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
        print("Back to the model!")
        return {'messages': results, 'flow_state': 'start'}
    
    
def get_response(employer_ID, user_input, state):
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
    router_system_prompt ='''
                            You are a routing agent tasked with determining the user's intent based on their message. Carefully analyze the user's input and select exactly one of the following paths based on the described criteria:

                        1. **Recruiter**:
                        - Choose this path if the user intends to find, filter, search, or retrieve candidates.
                        - Indications include queries about candidate profiles, candidate sourcing, questions about candidate attributes, skills, experience, or any recruitment-related information.

                        2. **Job Post Maker**:
                        - Choose this path if the user wishes to create, modify, enhance, or draft a job posting.
                        - Indications include requests to write job descriptions, job requirements, role summaries, responsibilities, or improve existing job listings.

                        3. **Meeting Scheduler**:
                        - Choose this path if the user's intent is to schedule, reschedule, cancel, or manage meetings or appointments.
                        - It can be date, time or an email address which is supposed to be invited to meeting.
                        - Indications include mentioning dates, times, scheduling conflicts, invitations, calendar management, or organizing meetings.

                        4. **General Questions**:
                        - Choose this path if the user's intent involves general inquiries about the company, policies, procedures, information about teams, projects, products, or any other data not specifically covered by recruitment, job posting, or meeting scheduling.

                        Based on the user's message, output ONLY one of these four strings without any additional text:
                        - recruiter
                        - job_post_maker
                        - meeting_scheduler
                        - general_questions

                        IMPORTANT:
                        If unsure, unclear, or unable to match precisely, pay attention to previous sentences or context to make the best decision.
    ''' 
    
    job_post_prompt = f"employer id is {employer_ID} /n/n"+ os.getenv('job_post_system_prompt')
    scheduler_prompt = os.getenv('SCHEDULER_AGENT_PROMPT')
    general_prompt = os.getenv('General_AGENT_PROMPT')

    model = ChatOpenAI(model="gpt-4o-2024-08-06", 
                       temperature=0, 
                       api_key= os.getenv("OPENAI_API_KEY"))
    
    router_model = ChatOpenAI(model="gpt-3.5-turbo", 
                       temperature=0, 
                       api_key= os.getenv("OPENAI_API_KEY"))
    
    tools = [search_candidate, calculate_free_times, schedule_meeting, job_creator]

    # messages = conversation_history + [HumanMessage(content=user_input)]
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, router_model, tools, checkpointer=memory,
                      system=prompt, user_input=user_input, router_prompt= router_system_prompt,
                        job_post_prompt=job_post_prompt, scheduler_prompt=scheduler_prompt, general_prompt=general_prompt)
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
    user_message: str = Form(...),  # Chat history as a string
    user_id: str = Form(...),  # User ID
    employer_id: Optional[str] = Form(None),  # Optional employer ID
):
    """
    - `agent_state`: String containing the chat history (JSON format).
    """
    try:
        # Check Redis for existing state
        stored_state = redis_client.hgetall(user_id)
        user_input = user_message
        user_message = HumanMessage(content=user_message)

        if stored_state:
            # Deserialize existing state
            state_dict = messages_from_dict(json.loads(stored_state['messages']))
            state_dict.append(user_message)
            agent_state = {
                "messages": state_dict,
                "flow_state": stored_state["flow_state"]
            }
        else:
            # Initialize new state if user does not exist
            agent_state = {"messages": [user_message], "flow_state": "start"}


        # Call your multi-agent system here
        updated_agent_state = get_response(employer_id, user_input, agent_state)

        # Serialize new state
        serialized_state = {
            "messages": json.dumps(messages_to_dict(updated_agent_state["messages"])),
            "flow_state": updated_agent_state.get("flow_state", None)
        }

         # Store serialized state in Redis hash
        redis_client.hmset(user_id, serialized_state)

        # Return the last generated message
        last_message = updated_agent_state["messages"][-1].content
        return JSONResponse(content={"user_id": user_id, "response": last_message})
    
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
    