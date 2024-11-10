import os
import openai
from openai import OpenAI
import streamlit as st
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
sys.setrecursionlimit(2000)
import time

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
        
        # Set metadata based on the source type (URL or PDF)
        metadata = {
            'url': content.get('url', '') if content.get('url') else source_name,  # Use source_name if URL is missing
            'title': content.get('title', source_name)  # Use source_name as title if title is missing
        }
        
        doc = Document(page_content=full_text, metadata=metadata)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        

        # Split document into chunks
        splits = text_splitter.split_documents([doc])
        

        if splits:
            time.sleep(2)
            # Add chunks to the vector database
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
    with fitz.open("pdf", file.read()) as pdf:  # Specify "pdf" as the file type and read the bytes
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            pdf_text += page.get_text("text") + "\n"
    return pdf_text


# Function to generate a unique directory name
def get_persist_directory(url, depth, max_url, pdf_files=None, source="web"):
    if url:
        # Hash the URL and PDF filenames together
        hash_input = url + "".join(sorted([file.name for file in pdf_files])) if pdf_files else url
        url_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        persist_directory = os.path.join('docs', 'chroma', f'{url_hash}_{source}_depth_{depth}_maxurl_{max_url}')
        return persist_directory
    else:
        # Hash the URL and PDF filenames together
        hash_input = "_" + "".join(sorted([file.name for file in pdf_files])) if pdf_files else url
        url_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        persist_directory = os.path.join('docs', 'chroma', f'{url_hash}_{source}_depth_{depth}_maxurl_{max_url}')
        return persist_directory


def main():
    st.title("AI Chatbot to interact with Web and PDF Data")

    # Sidebar inputs for URL, depth, max_url, language selection, and file upload
    st.sidebar.header("Configuration")
    url = st.sidebar.text_input("Enter the URL to scrape:", value=None)
    depth = st.sidebar.number_input("Enter the scraping depth:", min_value=0, max_value=10, value=0, step=1)
    max_url = st.sidebar.number_input("Enter the maximum number of URLs to crawl:", min_value=1, max_value=1000, value=10, step=1)
    language = st.sidebar.selectbox("Select Language:", ["English", "German"])
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

    # Initialize 'vectordb' in 'st.session_state' if not already
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None

    # Button to start processing
    if st.sidebar.button("Process URL and PDFs"):
        with st.spinner("Processing... This may take a while for larger depths."):
            persist_directory = get_persist_directory(url, depth, max_url, pdf_files=uploaded_files, source="web_pdf")

            embedding = CustomEmbeddingFunction('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)

            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                st.success(f"Loading existing data for {url} with uploaded PDFs at depth {depth} and max_url {max_url}.")
                vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            else:
                st.info(f"No existing data found for {url} with uploaded PDFs. Starting extraction...")
                vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)

                if url:
                    extractor = WebDataExtractor(url, max_depth=depth, max_url=max_url)
                    extractor.start_crawling(vectordb)
                    

                if uploaded_files:
                    for file in uploaded_files:
                        pdf_text = process_pdf(file)
                        pdf_content = {'title': file.name, 'full_text': pdf_text}
                        WebDataExtractor.process_and_store_content(pdf_content, vectordb, source_name=file.name)
                    

                st.success("Data extraction and processing completed.")
                

            st.session_state.vectordb = vectordb
            

    run_chatbot(language=language)




def run_chatbot(language):
    st.subheader("Chat with the AI Assistant")
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        st.error("API key not found in environment variables. Please set 'OPENAI_API_KEY'.")
        st.stop()

    chatbot = OpenAI(api_key=api_key)
    if language == 'German':
        system_msg = (
            "Ihre Sprache ist Deutsch"
            "Du bist ein deutscher KI-Assistent, der Benutzerfragen basierend auf dem bereitgestellten Kontext und Ihrem eigenen Wissen beantwortet. "
            "Jedes Kontextelement kann am Ende eine Quell-URL enthalten. Verwenden Sie beim Antworten zunächst die im Kontext bereitgestellten Informationen. "
            "Wenn der Kontext die Antwort nicht enthält, verwenden Sie Ihr eigenes Wissen, um eine hilfreiche Antwort zu geben. "
            "Fügen Sie die Quell-URLs in Ihre Antwort ein, wenn sie relevant sind. "
            "Seien Sie in Ihren Antworten genau, klar und präzise."
            f"Antworten Sie nur in Deutsch"

            )
    elif language == 'English':
        
        # Initialize system message for the chatbot
        system_msg = (
            "You are an english AI assistant that helps answer user questions based on the provided context and your own knowledge. "
            "Each piece of context may include a source URL at the end. When answering, first use the information provided "
            "in the context. If the context does not contain the answer, then use your own knowledge to provide a helpful response. "
            "Include the source URLs in your answer if they are relevant. "
            "Be accurate, clear, and concise in your answers."
            f"Respond only in english"
        )

    if "conversation" not in st.session_state:
        st.session_state.conversation = [{"role": "system", "content": system_msg}]

    def get_chatbot_response(user_input, Chatbot=chatbot):
        if st.session_state.vectordb is None:
            st.error("Please process a URL or PDF first by clicking 'Process URL and PDFs' in the sidebar.")
            return None

        vectordb = st.session_state.vectordb
        docs = vectordb.max_marginal_relevance_search(user_input, k=4, fetch_k=6)

        context = "\n\n".join([f"{doc.page_content}\nSource URL: {doc.metadata.get('url', '')}" for doc in docs])
        messages = st.session_state.conversation.copy()
        user_message = {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        messages.append(user_message)

        response = Chatbot.chat.completions.create(
            model="gpt-4o-2024-08-06", messages=messages
        )
        st.session_state.conversation.append({"role": "user", "content": user_input})
        bot_message = {"role": "assistant", "content": response.choices[0].message.content}
        st.session_state.conversation.append(bot_message)
        return bot_message["content"]

    with st.form(key="chat_form"):
        user_input = st.text_input("You:", key="user_input")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        chatbot_response = get_chatbot_response(user_input, Chatbot=chatbot)
        if chatbot_response:
            st.write(f"**Chatbot:** {chatbot_response}")

    st.write("### Conversation")
    for message in st.session_state.conversation[1:]:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Chatbot:** {message['content']}")


if __name__ == "__main__":
    main()
