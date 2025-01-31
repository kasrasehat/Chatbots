from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot Service API",
    description="A simple API for interacting with OpenAI GPT chatbot.",
    version="1.0.0"
)

# OpenAI API configuration
API_KEY = "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"
client = OpenAI(api_key=API_KEY)


# Route to redirect root to API documentation
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

# Main chatbot endpoint
@app.post("/chat")
async def chat_endpoint(
    user_prompt: str = Form(...),  # User prompt
):
    """
    Chat endpoint to interact with OpenAI GPT.
    - `user_prompt`: Input from the user.
    """
    try:
        system_prompt = "You are a helpful assistant."  
        # Construct the message payload
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

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
    uvicorn.run(app, host="0.0.0.0", port=8001)
