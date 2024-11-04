import os
import openai
import sys
sys.path.append('../..')
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
# _ = load_dotenv(find_dotenv()) # read local .env file

embedding = OpenAIEmbeddings()
openai.api_key = os.environ['OPENAI_API_KEY']

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("E:/articles/face-restoration/2101.04061v2.pdf"),
    PyPDFLoader("E:/articles/face-restoration/2205.06803v3.pdf"),
    PyPDFLoader("E:/articles/face-restoration/2312.15736v2.pdf")

]
docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

splits = text_splitter.split_documents(docs)
persist_directory = 'docs/chroma/'
# !rm -rf ./docs/chroma  # remove old database files if any
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())
question = 'what is the configuration like gpu used to train bfrffusion?'
docs = vectordb.max_marginal_relevance_search(question,k=3, fetch_k=6)
print(docs[0].page_content)

