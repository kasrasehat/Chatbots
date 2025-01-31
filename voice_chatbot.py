import openai
import os
import gradio as gr
from gtts import gTTS
import whisper
from tempfile import NamedTemporaryFile
from openai import OpenAI

# Initialize OpenAI API with your API key
api_key = "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" 

if not api_key:
    print("Error: API key not found in environment variables")
else:
    openai.api_key = api_key

chatbot = OpenAI(api_key=api_key)    

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize conversation memory
conversation_memory = [
    {"role": "system", "content": "You are a voice assistant of Berry On Mars company. when you start your conversation introduce yourself then answer the user"}
]

# Function to perform speech-to-text using Whisper
def speech_to_text(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result['text']

# Function to get chatbot response
def get_chatbot_response(user_input):
    # Append user input to conversation memory
    conversation_memory.append({"role": "user", "content": user_input})

    # Generate response
    response = chatbot.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=conversation_memory
    )

    # Append chatbot response to conversation memory
    chatbot_response = response.choices[0].message.content
    conversation_memory.append({"role": "assistant", "content": chatbot_response})

    return chatbot_response

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    temp_audio = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Define the Gradio Blocks interface
def get_demo():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>AI Chatbot</h1>")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources="microphone", type="filepath", label="Speak your query")
                submit_button = gr.Button("Submit", interactive=False)  # Initially disabled

            with gr.Column():
                user_text_output = gr.Textbox(label="Your Speech as Text")
                chatbot_response_output = gr.Textbox(label="Chatbot Response")
                chatbot_audio_output = gr.Audio(label="Chatbot Voice", autoplay=True)

        # Define the callback function
        def handle_interaction(audio):
            user_input = speech_to_text(audio)
            response = get_chatbot_response(user_input)
            chatbot_audio = text_to_speech(response)
            return user_input, response, chatbot_audio, gr.update(value=None)

        # Enable the submit button only when audio is uploaded
        def enable_submit(audio):
            if audio:
                return gr.update(interactive=True)
            return gr.update(interactive=False)

        audio_input.change(
            fn=enable_submit,
            inputs=[audio_input],
            outputs=[submit_button]
        )

        submit_button.click(
            fn=handle_interaction,
            inputs=[audio_input],
            outputs=[user_text_output, chatbot_response_output, chatbot_audio_output, audio_input]
        )

    return demo

if __name__ == "__main__":
    demo = get_demo()
    demo.launch(server_name="0.0.0.0", server_port=8701)
