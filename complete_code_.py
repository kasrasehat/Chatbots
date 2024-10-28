import os
import openai
import streamlit as st
import urllib.parse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import hashlib
from openai import OpenAI

# Define the WebDataExtractor class
class WebDataExtractor:
    def __init__(self, base_url, max_depth=2):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited_urls = set()
        self.extracted_data = []

    def is_valid_url(self, url):
        # Check if URL is in the same domain as the base URL
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

                # Process elements in natural order as they appear in the <body>
                body_content = soup.find('body')
                if body_content:
                    for element in body_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'], recursive=True):
                        text = element.get_text(strip=True)
                        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            headers.append(text)
                            # Add header to full_text with a separator
                            full_text.append(f"\n\n{text}")
                        elif element.name == 'p':
                            paragraphs.append(text)
                            # Add paragraph to full_text with a separator
                            full_text.append(f"\n{text}")

                content = {
                    'url': url,
                    'title': soup.title.string if soup.title else 'No Title',
                    'text': " ".join(paragraphs),
                    'headers': headers,
                    'images': [img['src'] for img in soup.find_all('img') if img.get('src')],
                    'full_text': "".join(full_text).strip()  # Join and strip leading/trailing newlines
                }
                return content
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
        return None

    def crawl(self, url, depth=0):
        if url in self.visited_urls or depth > self.max_depth:
            return
        self.visited_urls.add(url)

        print(f"Crawling: {url} at depth {depth}")
        content = self.extract_content(url)
        if content:
            self.extracted_data.append(content)

        if depth < self.max_depth:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        absolute_url = urljoin(url, link['href'])
                        if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                            self.crawl(absolute_url, depth + 1)
            except requests.RequestException as e:
                print(f"Error crawling {url}: {e}")

    def start_crawling(self):
        self.crawl(self.base_url)

    def get_data(self):
        return self.extracted_data

# Function to generate a unique directory name
def get_persist_directory(url, depth):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    persist_directory = os.path.join('docs', 'chroma', f'{url_hash}_depth_{depth}')
    return persist_directory

def main():

    st.title("AI Chatbot with Web Data Extraction")

    # Sidebar inputs for URL and depth
    st.sidebar.header("Configuration")
    url = st.sidebar.text_input("Enter the URL to scrape:", value="https://www.kindermann.de/en/")
    depth = st.sidebar.number_input("Enter the scraping depth:", min_value=0, max_value=10, value=0, step=1)

    # Initialize 'vectordb' in 'st.session_state' if not already
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None

    # Button to start processing
    if st.sidebar.button("Process URL"):
        with st.spinner("Processing... This may take a while for larger depths."):
            # Generate the unique directory
            persist_directory = get_persist_directory(url, depth)

            # Initialize embeddings
            embedding = OpenAIEmbeddings()

            # Check if the vector database already exists
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                st.success(f"Loading existing data for {url} at depth {depth}.")
                # Load the existing vector database
                vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            else:
                st.info(f"No existing data found for {url} at depth {depth}. Starting extraction...")
                # Initialize the extractor
                extractor = WebDataExtractor(url, max_depth=depth)
                extractor.start_crawling()

                # Get the extracted data
                data = extractor.get_data()

                # Prepare documents with metadata
                documents = []
                for item in data:
                    full_text = item.get('full_text', '')
                    if not full_text.strip():
                        continue
                    metadata = {
                        'url': item.get('url', ''),
                        'title': item.get('title', '')
                    }
                    doc = Document(page_content=full_text, metadata=metadata)
                    documents.append(doc)

                if not documents:
                    st.error("No documents to process. Please try a different URL or depth.")
                    st.stop()

                # Initialize text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=450,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ".", " ", ""]
                )

                # Split documents into chunks
                chunks = []
                for doc in documents:
                    splits = text_splitter.split_documents([doc])
                    chunks.extend(splits)

                if not chunks:
                    st.error("No chunks created from documents. Please try a different URL or depth.")
                    st.stop()

                # Create the vector database and persist it
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding,
                    persist_directory=persist_directory
                )
                # vectordb.persist()
                st.success(f"Data extraction and processing completed for {url} at depth {depth}.")

            # Store 'vectordb' in session state
            st.session_state.vectordb = vectordb

    # Proceed to the chatbot
    run_chatbot()

def run_chatbot():
    st.subheader("Chat with the AI Assistant")
    
    # Set your OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        st.error("API key not found in environment variables. Please set 'OPENAI_API_KEY'.")
        st.stop()
    # Initialize chatbot instance
    chatbot = OpenAI(api_key=api_key)
    # Initialize system message for the chatbot
    system_msg = (
        "You are an AI assistant that helps answer user questions based on the provided context and your own knowledge. "
        "When answering, first use the information provided in the context. If the context does not contain the answer, "
        "then use your own knowledge to provide a helpful response. Respond in the same language that the user is using. "
        "Be accurate, clear, and concise in your answers."
    )

    # Initialize conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = [{"role": "system", "content": system_msg}]

    # Function to get chatbot response
    def get_chatbot_response(user_input):
        # Check if 'vectordb' is available
        if st.session_state.vectordb is None:
            st.error("Please process a URL first by clicking 'Process URL' in the sidebar.")
            return None

        vectordb = st.session_state.vectordb

        # Retrieve relevant documents from vectordb
        docs = vectordb.max_marginal_relevance_search(user_input, k=3, fetch_k=6)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prepare messages for OpenAI API call
        messages = st.session_state.conversation.copy()

        # Append the user's message, including the context
        user_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_input}"
        }
        messages.append(user_message)

        # Call OpenAI API to generate response
        response = chatbot.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages)

        # Append messages to conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})
        bot_message = {"role": "assistant", "content": response.choices[0].message.content}
        st.session_state.conversation.append(bot_message)
        return bot_message["content"]

    # Streamlit UI
    with st.form(key="chat_form"):
        user_input = st.text_input("You:", key="user_input")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        # Get chatbot response and update the conversation
        chatbot_response = get_chatbot_response(user_input)
        if chatbot_response:
            st.write(f"**Chatbot:** {chatbot_response}")

    # Display conversation history
    st.write("### Conversation")
    for message in st.session_state.conversation[1:]:  # Skip the system message
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Chatbot:** {message['content']}")

if __name__ == "__main__":
    main()
