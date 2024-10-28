# chatbot.py

import os

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Assign environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ[
    'LANGCHAIN_API_KEY'] = ''  # Replace with your actual API key
os.environ['LANGCHAIN_PROJECT'] = 'sample_app'
os.environ['OPENAI_API_KEY'] = ''


def deep_scrape_website(start_url, max_depth=2, max_pages=50):
    visited_urls = set()
    scraped_data = []

    def is_valid_url(url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def scrape_page(url, depth):
        if url in visited_urls or len(scraped_data) >= max_pages:
            return

        visited_urls.add(url)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text from paragraphs
            text_content = 'page title: ' + soup.title.string + '\n'
            text_content += 'page link: ' + response.url + '\n'

            text_content += ' '.join([p.get_text() for p in soup.find_all('p')])

            scraped_data.append({"url": url, "content": text_content})

            if depth < max_depth:
                links = soup.find_all('a', href=True)
                valid_links = [urljoin(url, link['href']) for link in links
                               if is_valid_url(urljoin(url, link['href']))]

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(scrape_page, link, depth + 1)
                               for link in valid_links]
                    for future in as_completed(futures):
                        future.result()  # This will raise any exceptions that occurred

        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")

    scrape_page(start_url, 0)
    return scraped_data


def create_documents_from_scraped_data(scraped_data):
    if not scraped_data:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = [Document(page_content=chunk, metadata={"source": scraped_data["url"]})
            for chunk in splitter.split_text(scraped_data["content"])]
    return docs


def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def create_chatbot(vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""you are an employer from kindermann.de and you 
    Answer the questions regarding the website https://www.kindermann.de/ you will also have a context to help you answering questions
    do not write sentences like As an employer from Kindermann GmbH to start just start with 'We'
    Answer question only based on the Context,If you get questions other than your company or its website, answer you do not know anything about this topic.

    Context: {context}
    Question: {input}

    Answer: """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def ask_question(chain, query):
    response = chain.invoke({"input": query})
    return response['answer']


# Main function to integrate all steps
if __name__ == "__main__":
    url = "https://www.kindermann.de/"  # Replace with your target website
    scraped_data_list = deep_scrape_website(url)

    all_documents = []
    for scraped_data in scraped_data_list:
        documents = create_documents_from_scraped_data(scraped_data)
        all_documents.extend(documents)

    if all_documents:
        vectorstore = create_vector_store(all_documents)
        chatbot_chain = create_chatbot(vectorstore)

        # Example query
        question = "can you tell me more about AV-over-IP in your company?"
        answer = ask_question(chatbot_chain, question)
        print(f"---------------------------------------------------------------------")
        print(f"Answer: {answer}")
    else:
        print("Failed to scrape any content from the website.")