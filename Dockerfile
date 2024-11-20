# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Gradio will run on
EXPOSE 8700

# Set environment variables for OpenAI API key (adjust as needed)
# ENV OPENAI_API_KEY=your_openai_api_key_here

# Command to run the Gradio application
CMD ["python", "gradio_completed_code.py"]
