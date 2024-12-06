import numpy as np
from tavily import TavilyClient
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
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool 

sys.setrecursionlimit(2000)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# Define tools for the flow

@tool
def data_retriever():
    """
    Retrieves related data using RAG system or TavilySearch.
    """
    return "This tool uses RAG (retrieval augmented generation) and TavilySearch to find and retrieve related data."
    
    
    
@tool
def ask_email():
    """
    This tool prompts the user to provide their email address.
    """
    return "Please provide your email address."

@tool
def check_email_format(email):
    """
    This tool verifies if the provided email address is valid and exists in database or not.
    Args:
        email (str): The email address provided by the user.
    Returns:
        str or None: The email address if valid, otherwise None.
    """
    if not email or "@" not in email:
        return 'email is incorrect'  
    return 'email is correct' 

@tool
def send_code(email):
    """
    This tool sends a verification code to the verified provided email address.
    Args:
        email (str): The email address to which the code is sent.
    Returns:
        str: A message indicating that the code has been sent.
    """
    return f"Verification code has been sent to {email}. Please enter the code."

@tool
def verify_email(code):
    """
    This tool verifies if the code provided by the user is correct and same as the one sent to email address or not.
    Args:
        code (str): The verification code provided by the user.
    Returns:
        str: A message indicating whether the code is correct or incorrect.
    """
    if str(code) == '1234':  # Assume 1234 is the correct code for testing purposes
        return "code is same with the one sent to email"
    return "code is incorrect"
    

@tool
def ask_password(args=None):
    """
    This tool prompts the user to enter their new password for first time.
    """
    return "Please type your new password for first time."

@tool
def retype_password(args=None):
    """
    This tool prompts the user to retype password provided earlier for second time for confirmation.
    It has to be called immediately after ask_password().
    """
    return "Please retype your password for confirmation."

@tool
def conform_passwords(state: AgentState):
    """
    This tool confirms that both passwords requested from user are identical.
    It uses the last two human messages from the agent state.
    Args:
        state (AgentState): The current state of the agent containing messages.
    Returns:
        str: A message indicating whether the passwords match or not.
    """
    human_messages = [msg.content for msg in state['messages'] if isinstance(msg, HumanMessage)][-2:]
    if len(human_messages) == 2 and human_messages[0] == human_messages[1]:
        return "Passwords match."
    return "Passwords do not match. Please try again."

@tool
def record_password():
    """
    This tool records the new password.

    Returns:
        str: A message indicating whether the password change was successful.
    """
    return "Password has been changed successfully."


class Agent:
    def __init__(self, vectordb , model, tools, checkpointer, system="", user_input='', language='English'):
        self.system = system
        self.vectordb = vectordb
        self.user_input = user_input
        self.language  = language
        self.model_rag = model
        
        graph = StateGraph(AgentState)
        
        # Graph nodes
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_node("search_data", self.call_rag)
        
        # Conditional edges and flow control
        graph.add_conditional_edges("llm", self.exists_action, {'take_action': "action", 'retrieve_data': "search_data", False: END})
        graph.add_edge("action", "llm")
        graph.add_edge("search_data", "llm")
        graph.set_entry_point("llm")
        
        self.graph = graph.compile(checkpointer=checkpointer)  # Use the checkpointer passed as a parameter
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            message = self.model.invoke(messages)
        return {'messages': [message]}


    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        if len(result.tool_calls) > 0:
            
            if result.tool_calls[0]['name'] == 'data_retriever': 
                return 'retrieve_data'
            else: 
                return 'take_action'
        else: 
            return False

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t['name']
            tool_args = t['args']
            print(f"Calling: {tool_name} with args: {tool_args}")
            result = self.tools[tool_name](*tool_args if isinstance(tool_args, list) else [tool_args])
            results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    
    def call_rag(self, state: AgentState):
        
        similarity_threshold = 0.65  # Set a value between 0 and 1

        # Perform the initial search to get documents (you can still use max_marginal_relevance_search)
        docs = self.vectordb.max_marginal_relevance_search(self.user_input, k=6, fetch_k=8)

        # Get the embedding of the user query
        embedding_function = CustomEmbeddingFunction('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        user_embedding = embedding_function.embed_query(self.user_input)

        # Filter documents based on cosine similarity
        filtered_docs = []

        for doc in docs:
            # Get the document embedding
            doc_embedding = embedding_function.embed_query(doc.page_content)  # You may use `embed_documents` if you have multiple pages
            
            # Calculate cosine similarity
            similarity = np.dot(user_embedding, doc_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(doc_embedding))
            
            # Apply the similarity threshold
            if similarity >= similarity_threshold:
                filtered_docs.append(doc)
        
        if len(filtered_docs)==0:

            # Step 1. Instantiating your TavilyClient
            tavily_client = TavilyClient(api_key="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")

            # Step 2. Executing a context search query
            context = tavily_client.get_search_context(query=self.user_input)
            tool_calls = state['messages'][-1].tool_calls
            t = tool_calls[0]
            tool_name = t['name']
            tool_args = t['args']
            results = []
            print(f"Calling: tavily serach with args: {self.user_input}")
            result = context
            results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
            print("Back to the model!")
            return {'messages': results}
        
        else:
            if self.language == 'German': 
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
            context = "\n\n".join([f"{doc.page_content}\nSource URL: {doc.metadata.get('url', '')}" for doc in docs])
            messages = conversation + [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}]

            response = self.model_rag.invoke(
                messages=messages
                )
            
            tool_calls = state['messages'][-1].tool_calls
            t = tool_calls[0]
            tool_name = t['name']
            tool_args = t['args']
            results = []
            print(f"Calling: RAG system with args: {self.user_input}")
            result = response.choices[0].message.content
            results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
            print("Back to the model!")
            return {'messages': results}
        

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

# # Use AgentState
# def user_agent_state():
#     state: AgentState = {
#         "messages": []
#     }
#     return state

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
                WebDataExtractor.process_and_store_content(pdf_content, vectordb, source_name=file.name.split('/')[-1])
        
        status_message += "<p style='text-align: center;'>Data extraction and processing completed.</p>"

    return vectordb, status_message

def get_response(user_input, vectordb, language, state):
    # This function now has access to the full conversation history from state
    # Construct a response based on state messages

    # Here you can interact with the LLM model using all past messages
    # Example: Generate a response using past conversation history
    conversation_history = state["messages"]

    # If using OpenAI or any LLM, you'd pass the full conversation history
    prompt = """
                You are a smart assistant tasked with guiding the user through changing their password or serach any data, if they want. 
                    You have the following tools at your disposal for changing password:
                        1. ask_email: Prompt the user to provide their email address.
                        2. check_email_format: Verify if the provided email address format is valid.
                        3. send_code: Send a verification code to the format-verified email address.
                        4. verify_email: check the code provided by the user is same as sent code to email.
                        5. ask_password: Ask the user to provide a new password for first time.
                        6. retype_password: Ask the user to retype new password for second time for confirmation.
                        7. conform_passwords: Check if the two passwords match.
                        8. record_password: Record the new password and finalize the process.
                    the order of these tools for changing password is important.                     
                    
                    you have the following tools at your disposal for searching data:
                        data_retrieve: Retrieve the data from the database or web.

                    When appropriate, call these tools to proceed through the steps. Do not answer questions by yourself as much as possible.
                    Guide the user step-by-step until the user achieve goal which can be the password change process or serach data.
                    if user provides wrong answer ask 2 times again. if he can not provide right answer, return to the first step which is asking for email address.
             """

    model = ChatOpenAI(model="gpt-4o-2024-08-06")
    tools = [ask_email, check_email_format, send_code, verify_email, ask_password, retype_password, conform_passwords, record_password, data_retriever]

    # messages = conversation_history + [HumanMessage(content=user_input)]
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(vectordb, model, tools, checkpointer=memory, system=prompt, user_input=user_input, language=language)
        result = abot.graph.invoke({"messages": conversation_history}, {"configurable": {"thread_id": "1"}})
        ai_response = result['messages'][-1].content
        
    return ai_response    


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
        # State to hold the agent messages
        agent_state = gr.State(value={"messages": []})

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

        def on_send_button(user_input, vectordb_state, agent_state, language):
            # Retrieve current state of messages
            state = agent_state  # This is the current agent state holding all previous messages

            # Append the new user input as a HumanMessage to state
            state['messages'].append(HumanMessage(content=user_input))

            # Use the updated state to get a response
            response_content = get_response(user_input, vectordb_state, language, state)

            # Append the model's response to the state
            response_message = AIMessage(content=response_content)
            state['messages'].append(response_message)

            # Return the updated agent state and the response for the UI
            return response_content, state

        # Link Buttons to Handlers
        process_button.click(
            fn=on_process_button,
            inputs=[url_input, depth_slider, max_url_slider, language_radio, uploaded_files],
            outputs=[vectordb_state, processing_status, user_input, process_button, send_button]
        )

        send_button.click(
            fn=on_send_button,
            inputs=[user_input, vectordb_state, agent_state, language_radio],
            outputs=[chat_output, agent_state]
        )

    return demo

if __name__ == "__main__":
    demo = get_demo()
    demo.launch(server_name="0.0.0.0", server_port=8701)