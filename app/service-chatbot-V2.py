from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import traceback
import json
import re

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot Service API",
    description="A simple API for interacting with OpenAI GPT chatbot.",
    version="1.0.0"
)

# OpenAI API configuration
API_KEY = "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"
client = OpenAI(api_key=API_KEY)

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
@app.post("/chat")
async def chat_endpoint(
    chat_history: str = Form(...)  # Chat history as a string
):
    """
    Chat endpoint to interact with OpenAI GPT.
    - `chat_history`: String containing the chat history (JSON format).
    """
    try:
        system_prompt = {'role': 'system', 'content': 'You are a helpful assistant.'}

        # Convert the chat_history string to a list
        try:
            # Fix JSON using regex
            # Replace single quotes with double quotes to make it valid JSON
            stringified_data = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', chat_history)
            chat_history_list = json.loads(stringified_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for chat history.")

        # Validate that chat_history_list is a list of dictionaries with 'role' and 'content'
        if not isinstance(chat_history_list, list) or not all(
            isinstance(message, dict) and 'role' in message and 'content' in message for message in chat_history_list
        ):
            raise HTTPException(status_code=400, detail="Chat history must be a list of messages with 'role' and 'content' keys.")

        # Add the system prompt to the beginning of the history
        messages = [system_prompt] + chat_history_list

        # Send request to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0
        )

        # Extract and return the response
        assistant_response = response.choices[0].message.content
        return JSONResponse(content={"response": assistant_response})

    except Exception as e:
        # Handle errors gracefully
        full_traceback = traceback.format_exc()
        return JSONResponse(
            content={"error": str(e), "traceback": full_traceback},
            status_code=500
        )

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8002)
    
    
    
    
