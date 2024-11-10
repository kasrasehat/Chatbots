import os
import openai
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
from openai import OpenAI
import sys
from sentence_transformers import SentenceTransformer
sys.setrecursionlimit(2000)


# Custom embedding function to wrap SentenceTransformer
class CustomEmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Use SentenceTransformer's `encode` method to embed a list of documents and convert to list format
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        # For a single query embedding, convert to list format
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()


# Define the WebDataExtractor class
class WebDataExtractor:
    def __init__(self, base_url, max_depth=2, max_url=10):
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_url = max_url
        self.visited_urls = set()

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

    @staticmethod
    def process_and_store_content(content, vectordb):
        full_text = content.get('full_text', '')
        if not full_text.strip():
            return
        metadata = {
            'url': content.get('url', ''),
            'title': content.get('title', '')
        }
        doc = Document(page_content=full_text, metadata=metadata)

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Split document into chunks
        splits = text_splitter.split_documents([doc])

        if splits:
            # Add chunks to the vector database
            vectordb.add_documents(splits)
        else:
            print(f"No chunks created for {content.get('url', '')}")

    def crawl(self, url, vectordb, depth=0):
        if url in self.visited_urls or depth > self.max_depth or len(self.visited_urls) >= self.max_url:
            return
        self.visited_urls.add(url)

        print(f"Crawling: {url} at depth {depth}")
        content = self.extract_content(url)
        if content:
            # Process the content: chunk and store in vectordb
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


# Function to generate a unique directory name
def get_persist_directory(url, depth, max_url):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    persist_directory = os.path.join('docs', 'chroma', f'{url_hash}_depth_{depth}_maxurl_{max_url}')
    return persist_directory


def main():

    st.title("AI Chatbot with Web Data Extraction")

    # Sidebar inputs for URL, depth, max_url, and language selection
    st.sidebar.header("Configuration")
    url = st.sidebar.text_input("Enter the URL to scrape:", value="https://www.kindermann.de/en/")
    depth = st.sidebar.number_input("Enter the scraping depth:", min_value=0, max_value=10, value=0, step=1)
    max_url = st.sidebar.number_input("Enter the maximum number of URLs to crawl:", min_value=1, max_value=1000, value=10, step=1)

    # **Change #1: Language Selection Dropdown**
    language = st.sidebar.selectbox("Select Language:", ["English", "German"])

    # Initialize 'vectordb' in 'st.session_state' if not already
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None

    # Button to start processing
    if st.sidebar.button("Process URL"):
        with st.spinner("Processing... This may take a while for larger depths."):
            # Generate the unique directory
            persist_directory = get_persist_directory(url, depth, max_url)

            # **Change #2: Initialize embeddings based on selected language**
            embedding = CustomEmbeddingFunction('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

            # Ensure the persist_directory exists
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)

            # Initialize or load the vector database
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                st.success(f"Loading existing data for {url} at depth {depth} with max_url {max_url}.")
                # Load the existing vector database
                vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            else:
                st.info(f"No existing data found for {url} at depth {depth} with max_url {max_url}. Starting extraction...")
                # Initialize an empty vector database
                vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)

                # Initialize the extractor
                extractor = WebDataExtractor(url, max_depth=depth, max_url=max_url)
                extractor.start_crawling(vectordb)

                st.success(f"Data extraction and processing completed for {url} at depth {depth} with max_url {max_url}.")

            # Store 'vectordb' in session state
            st.session_state.vectordb = vectordb

    # Proceed to the chatbot
    run_chatbot(language=language)


def run_chatbot(language):
    st.subheader("Chat with the AI Assistant")

    # Set your OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        st.error("API key not found in environment variables. Please set 'OPENAI_API_KEY'.")
        st.stop()

    # Initialize chatbot instance
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

    # Initialize conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = [{"role": "system", "content": system_msg}]

    # Function to get chatbot response
    def get_chatbot_response(user_input, Chatbot=chatbot):
        # Check if 'vectordb' is available
        if st.session_state.vectordb is None:
            st.error("Please process a URL first by clicking 'Process URL' in the sidebar.")
            return None

        vectordb = st.session_state.vectordb

        # Retrieve relevant documents from vectordb
        docs = vectordb.max_marginal_relevance_search(user_input, k=4, fetch_k=6)

        # Extract context with URLs appended
        context = ""
        for doc in docs:
            doc_content = doc.page_content
            url = doc.metadata.get('url', '')
            if url:
                doc_content += f"\nSource URL: {url}"
            context += doc_content + "\n\n"

        # # Append the user's message to conversation history
        # st.session_state.conversation.append({"role": "user",
        #                                       "content": user_input})

        # # Prepare messages for OpenAI API call
        # messages = st.session_state.conversation.copy()

        # # Append the context to the assistant's last message
        # messages.append({
        #     "role": "assistant",
        #     "content": f"Context:\n{context}\n\n"
        # })

        # # Call OpenAI API to generate response
        # response = Chatbot.chat.completions.create(
        #     model="gpt-4o-2024-08-06",  # or "gpt-4" if you have access
        #     messages=messages
        # )

        # # Append the assistant's response to the conversation
        # bot_message = {"role": "assistant", "content": response.choices[0].message.content}
        # st.session_state.conversation.append(bot_message)

        # Prepare messages for OpenAI API call
        messages = st.session_state.conversation.copy()

        # Append the user's message, including the context
        user_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_input}"
        }
        messages.append(user_message)

        # Call OpenAI API to generate response
        response = Chatbot.chat.completions.create(
            model="gpt-4o-2024-08-06",  # or "gpt-4" if you have access
            messages=messages
        )

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
        chatbot_response = get_chatbot_response(user_input, Chatbot=chatbot)
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
