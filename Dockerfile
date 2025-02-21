# Use an official Python runtime as the base image
FROM python:3.12.3-slim

# Step 2: Set a working directory
WORKDIR /app

# Step 3: Copy the requirements.txt file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the application code into the container
COPY . /app

# Step 5: Expose the port on which your app will run
EXPOSE 8000

# Step 6: Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "Chatbot.service-recruiter-agent:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
