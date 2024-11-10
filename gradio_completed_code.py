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

sys.setrecursionlimit(2000)

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

    return vectordb, status_message


def get_response(user_input, vectordb, language):
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


def get_demo():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>AI Chatbot to interact with Website and PDF</h1>")

        # Configuration Inputs
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

        # Button Handlers
        def on_process_button(url, depth, max_url, language, uploaded_files):
            # Show processing message and disable buttons
            processing_status_text = "<p style='text-align: center;'>Processing... Please wait.</p>"
            user_input_interactive = False
            process_button_interactive = False
            send_button_interactive = False

            # Initialize vector database
            vectordb, status_message = initialize_data(url, depth, max_url, language, uploaded_files)

            # Update status and enable buttons
            processing_status_text = status_message
            user_input_interactive = True
            process_button_interactive = True
            send_button_interactive = True

            return vectordb, processing_status_text, gr.update(interactive=user_input_interactive), gr.update(interactive=process_button_interactive), gr.update(interactive=send_button_interactive)

        def on_send_button(user_input, vectordb_state, language):
            response = get_response(user_input, vectordb_state, language)
            return response

        # Link Buttons to Handlers
        process_button.click(
            fn=on_process_button,
            inputs=[url_input, depth_slider, max_url_slider, language_radio, uploaded_files],
            outputs=[vectordb_state, processing_status, user_input, process_button, send_button]
        )

        send_button.click(
            fn=on_send_button,
            inputs=[user_input, vectordb_state, language_radio],
            outputs=[chat_output]
        )

    return demo


if __name__ == "__main__":
    demo = get_demo()
    demo.launch(server_name="0.0.0.0", server_port=8900)
