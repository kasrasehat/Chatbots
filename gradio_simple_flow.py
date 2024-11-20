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
import numpy as np
import csv
import re
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

sys.setrecursionlimit(2000)
# Add this dictionary for flows
defined_flows = {
    "password_change": ["change my pass", "i lost my pass code", "I forgot my passcode","I forgot my password", "lost password", "reset my password", "password change", "how to recovery my passcode", "update password"],
    "account_creation": ["create account", "new account", "sign up", "register"],
    "billing_issue": ["billing problem", "payment issue", "charged incorrectly", "billing support"],
    # Add more flows as needed
}

# Custom embedding function to wrap SentenceTransformer
class CustomEmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Define the WebDataExtractor class
class WebDataExtractor:
    def __init__(self, base_url, max_depth=2, max_url=10):
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_url = max_url
        self.visited_urls = set()

    def is_valid_url(self, url):
        parsed_base_url = urlparse(self.base_url)
        parsed_url = urlparse(url)
        return parsed_url.netloc == parsed_base_url.netloc

    def extract_content(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                headers = []
                paragraphs = []
                full_text = []

                body_content = soup.find('body')
                if body_content:
                    for element in body_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'], recursive=True):
                        text = element.get_text(strip=True)
                        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            headers.append(text)
                            full_text.append(f"\n\n{text}")
                        elif element.name == 'p':
                            paragraphs.append(text)
                            full_text.append(f"\n{text}")

                content = {
                    'url': url,
                    'title': soup.title.string if soup.title else 'No Title',
                    'text': " ".join(paragraphs),
                    'headers': headers,
                    'images': [img['src'] for img in soup.find_all('img') if img.get('src')],
                    'full_text': "".join(full_text).strip()
                }
                return content
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
        return None

    @staticmethod
    def process_and_store_content(content, vectordb, source_name=None):
        full_text = content.get('full_text', '')
        if not full_text.strip():
            return
        
        metadata = {
            'url': content.get('url', '') if content.get('url') else source_name,
            'title': content.get('title', source_name)
        }
        
        doc = Document(page_content=full_text, metadata=metadata)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        splits = text_splitter.split_documents([doc])
        
        if splits:
            time.sleep(2)
            vectordb.add_documents(splits)
        else:
            print(f"No chunks created for {source_name or content.get('url', '')}")

    def crawl(self, url, vectordb, depth=0):
        if url in self.visited_urls or depth > self.max_depth or len(self.visited_urls) >= self.max_url:
            return
        self.visited_urls.add(url)

        print(f"Crawling: {url} at depth {depth}")
        content = self.extract_content(url)
        if content:
            self.process_and_store_content(content, vectordb)

        if depth < self.max_depth:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        absolute_url = urljoin(url, link['href'])
                        if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                            self.crawl(absolute_url, vectordb, depth + 1)
            except requests.RequestException as e:
                print(f"Error crawling {url}: {e}")

    def start_crawling(self, vectordb):
        self.crawl(self.base_url, vectordb)

def initialize_data(url, depth, max_url, language, uploaded_files):
    persist_directory = get_persist_directory(url, depth, max_url, pdf_files=uploaded_files, source="web_pdf")
    embedding = CustomEmbeddingFunction('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Ensure the directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # Status messages
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        status_message = f"<p style='text-align: center;'>Loading existing data for {url} with uploaded PDFs at depth {depth} and max_url {max_url}.</p>"
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        status_message = f"<p style='text-align: center;'>No existing data found for {url} with uploaded PDFs. Starting extraction...</p>"
        vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)

        if url:
            status_message += f"<p style='text-align: center;'>Crawling website data...</p>"
            extractor = WebDataExtractor(url, max_depth=depth, max_url=max_url)
            extractor.start_crawling(vectordb)
        
        if uploaded_files:
            for file in uploaded_files:
                status_message += f"<p style='text-align: center;'>Processing PDF: {file.name}</p>"
                pdf_text = process_pdf(file)
                pdf_content = {'title': file.name, 'full_text': pdf_text}
                WebDataExtractor.process_and_store_content(pdf_content, vectordb, source_name=file.name)
        
        status_message += "<p style='text-align: center;'>Data extraction and processing completed.</p>"

    return vectordb, status_message, embedding


# PDF processing function
def process_pdf(file):
    pdf_text = ""
    with fitz.open(file) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            pdf_text += page.get_text("text") + "\n"
    return pdf_text

# Generate a unique directory name
def get_persist_directory(url, depth, max_url, pdf_files=None, source="web"):
    if url:
        hash_input = url + "".join(sorted([file.name for file in pdf_files])) if pdf_files else url
        url_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        persist_directory = os.path.join('docs', 'chroma', f'{url_hash}_{source}_depth_{depth}_maxurl_{max_url}')
        return persist_directory
    else:
        hash_input = "_" + "".join(sorted([file.name for file in pdf_files])) if pdf_files else url
        url_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        persist_directory = os.path.join('docs', 'chroma', f'{url_hash}_{source}_depth_{depth}_maxurl_{max_url}')
        return persist_directory

# Retrieve user from CSV file
def retrieve_user(email, csv_file='/root/kasra/projects/Chatbots/users.csv'):
    try:
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Email'].lower() == email.lower():
                    return row
    except FileNotFoundError:
        print("CSV file not found.")
    return None

# Update user's password in CSV file
def update_user_password(email, new_password, csv_file='users.csv'):
    try:
        rows = []
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Email'].lower() == email.lower():
                    row['Password'] = new_password
                rows.append(row)

        with open(csv_file, mode='w', newline='') as file:
            fieldnames = ['Name', 'Family_Name', 'Email', 'Password']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except FileNotFoundError:
        print("CSV file not found.")

# Extract email from input
def extract_email_from_input(user_input):
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w{2,4}'
    match = re.search(email_pattern, user_input)
    return match.group(0) if match else None


def send_verification_email(to_email, verification_code):
    sender_email = "ks13.st13@gmail.com"
    sender_password = "137213Qqq"  # Use an App Password here
    subject = "Password Reset Verification Code"
    body = f"""
    Dear User,
    
    We are processing your password changing request.
    To safeguard your account and verify your identity, we've generated a verification code for you:
    {verification_code}
    
    The code is valid for 10 minutes.
    
    Please do not disclose this code to anyone to avoid potential major financial loss.
    
    If you don't recognize this activity, please contact our official customer support promptly.
    
    Thank you for your support!
    BerryOnMars Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
         
         # Create an SSL context
        context = ssl.create_default_context()
         # Connect to Gmail SMTP server using SSL
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# State to track password change flow
password_change_state = {
    "active": False,
    "step": 0,
    "email": None,
    "verification_code": None,
    "attempts": 0
}

# Calculate similarity between user input and defined flows
def calculate_similarity(user_input, embedding_function, defined_flows):
    user_embedding = embedding_function.embed_query(user_input)
    best_flow = None
    highest_similarity = -np.inf

    for flow, phrases in defined_flows.items():
        for phrase in phrases:
            phrase_embedding = embedding_function.embed_query(phrase)
            similarity = np.dot(user_embedding, phrase_embedding)/ (np.linalg.norm(user_embedding) * np.linalg.norm(phrase_embedding))

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_flow = flow

    return best_flow, highest_similarity

# Handle password change flow
def handle_password_change_flow(user_input):
    global password_change_state
    response = ""

    if password_change_state["step"] == 0:
        response = "The password changing process has been started. Please provide the email address used during sign-up."
        password_change_state["step"] = 1
        password_change_state["attempts"] = 0

    elif password_change_state["step"] == 1:
        email = extract_email_from_input(user_input)
        if email:
            user = retrieve_user(email)
            if user:
                password_change_state["email"] = email
                password_change_state["verification_code"] = 12345
                send_verification_email(email, password_change_state["verification_code"])
                response = f"A verification code has been sent to your email. Please enter the code."
                password_change_state["step"] = 2
            else:
                password_change_state["attempts"] += 1
                if password_change_state["attempts"] >= 3:
                    response = "No user found with the provided email after 3 attempts. Returning to main process."
                    password_change_state["active"] = False
                else:
                    response = "No user found with the provided email. Please try again."
        else:
            password_change_state["attempts"] += 1
            if password_change_state["attempts"] >= 3:
                response = "Invalid email format after 3 attempts. Returning to main process."
                password_change_state["active"] = False
            else:
                response = "Invalid email format. Please try again."

    elif password_change_state["step"] == 2:
        if user_input.isdigit() and int(user_input) == password_change_state["verification_code"]:
            response = "Please type your new password:"
            password_change_state["step"] = 3
        else:
            password_change_state["attempts"] += 1
            if password_change_state["attempts"] >= 3:
                response = "Failed to verify code after 3 attempts. Returning to main process."
                password_change_state["active"] = False
            else:
                response = "Incorrect verification code. Please try again."

    elif password_change_state["step"] == 3:
        password_change_state["new_password"] = user_input
        response = "Please type your new password again to confirm:"
        password_change_state["step"] = 4

    elif password_change_state["step"] == 4:
        if user_input == password_change_state["new_password"]:
            update_user_password(password_change_state["email"], user_input)
            response = "Password has been successfully changed. Returning to main process."
            password_change_state["active"] = False
        else:
            response = "Passwords do not match. The process has been terminated. Returning to main process."
            password_change_state["active"] = False

    return response

# Get response function
def get_response(user_input, vectordb, embedding_function, language):
    global password_change_state
    if password_change_state["active"]:
        return handle_password_change_flow(user_input)

    best_flow, similarity = calculate_similarity(user_input, embedding_function, defined_flows)
    similarity_threshold = 0.7

    if best_flow == "password_change" and similarity > similarity_threshold:
        password_change_state["active"] = True
        password_change_state["step"] = 0
        return handle_password_change_flow(user_input)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "API key not found in environment variables. Please set 'OPENAI_API_KEY'."

    chatbot = OpenAI(api_key=api_key)

    if language == 'German':
        system_msg = (
            "Ihre Sprache ist Deutsch. "
            "Du bist ein deutscher KI-Assistent, der Benutzerfragen basierend auf dem bereitgestellten Kontext und Ihrem eigenen Wissen beantwortet. "
            "Jedes Kontextelement kann am Ende eine Quell-URL enthalten. Verwenden Sie beim Antworten zunächst die im Kontext bereitgestellten Informationen. "
            "Wenn der Kontext die Antwort nicht enthält, verwenden Sie Ihr eigenes Wissen, um eine hilfreiche Antwort zu geben. "
            "Fügen Sie die Quell-URLs in Ihre Antwort ein, wenn sie relevant sind. "
            "Seien Sie in Ihren Antworten genau, klar und präzise."
        )
    else:
        system_msg = (
            "You are an english AI assistant that helps answer user questions based on the provided context and your own knowledge. "
            "Each piece of context may include a source URL at the end. When answering, first use the information provided "
            "in the context. If the context does not contain the answer, then use your own knowledge to provide a helpful response. "
            "Include the relative source URLs in your answer. "
            "Be accurate, clear, informative, and concise in your answers."
        )

    conversation = [{"role": "system", "content": system_msg}]
    docs = vectordb.max_marginal_relevance_search(user_input, k=4, fetch_k=6)
    context = "\n\n".join([f"{doc.page_content}\nSource URL: {doc.metadata.get('url', '')}" for doc in docs])
    messages = conversation + [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}]

    response = chatbot.chat.completions.create(
        model="gpt-4o-2024-08-06", messages=messages
    )
    return response.choices[0].message.content

# Update the Gradio demo to pass embedding_function to the on_send_button function
def get_demo():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>AI Chatbot to interact with Website and PDF</h1>")

        # Configuration Inputs
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(label="Enter the URL to scrape:")
                depth_slider = gr.Slider(0, 10, step=1, label="Enter the scraping depth:", value=1)
                max_url_slider = gr.Slider(1, 1000, step=1, label="Enter the maximum number of URLs to crawl:", value=10)
                language_radio = gr.Radio(["English", "German"], label="Select Language:", value="English")
                uploaded_files = gr.File(label="Upload PDF files", file_count="multiple", type="filepath")
                process_button = gr.Button("Process")
                processing_status = gr.Markdown("<p style='text-align: center;'>Click 'Process' to start.</p>")

            # Chat Interface
            with gr.Column():
                user_input = gr.Textbox(label="Your question:", placeholder="Type your question here...", interactive=False)
                chat_output = gr.Textbox(label="Chatbot Response", placeholder="Chatbot will respond here...", interactive=False)
                send_button = gr.Button("Send", interactive=False)

        # State to hold the vector database
        vectordb_state = gr.State()
        embedder_state = gr.State()

        # Button Handlers
        def on_process_button(url, depth, max_url, language, uploaded_files):
            # Show processing message and disable buttons
            processing_status_text = "<p style='text-align: center;'>Processing... Please wait.</p>"
            user_input_interactive = False
            process_button_interactive = False
            send_button_interactive = False

            # Initialize vector database
            vectordb, status_message, embedding = initialize_data(url, depth, max_url, language, uploaded_files)

            # Update status and enable buttons
            processing_status_text = status_message
            user_input_interactive = True
            process_button_interactive = True
            send_button_interactive = True

            return vectordb, processing_status_text, embedding, gr.update(interactive=user_input_interactive), gr.update(interactive=process_button_interactive), gr.update(interactive=send_button_interactive)

        def on_send_button(user_input, vectordb_state, embedder, language):
            response = get_response(user_input, vectordb_state, embedder, language)
            return response

        # Link Buttons to Handlers
        process_button.click(
            fn=on_process_button,
            inputs=[url_input, depth_slider, max_url_slider, language_radio, uploaded_files],
            outputs=[vectordb_state, processing_status, embedder_state, user_input, process_button, send_button]
        )

        send_button.click(
            fn=on_send_button,
            inputs=[user_input, vectordb_state, embedder_state, language_radio],
            outputs=[chat_output]
        )

    return demo

if __name__ == "__main__":
    demo = get_demo()
    demo.launch(server_name="0.0.0.0", server_port=8913)
