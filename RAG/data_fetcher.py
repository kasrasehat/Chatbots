from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain_community.document_loaders import YoutubeAudioLoader
import os
os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import NotionDirectoryLoader

loader = PyPDFLoader("E:/articles/face-restoration/bfrffusion.pdf")
pages = loader.load()
print(len(pages))
url = "https://www.youtube.com/watch?v=w55C8cLWz74"
save_dir="E:/codes_py/Chatbots/docs"
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
print(docs[0].page_content[:300])

# urls=["https://www.youtube.com/watch?v=w55C8cLWz74", "https://www.youtube.com/watch?v=w55C8cLWz74"]
# from langchain_community.document_loaders import UnstructuredURLLoader
# loader = UnstructuredURLLoader(urls, save_dir)
# data = loader.load()



loader = WebBaseLoader("https://github.com/kasrasehat/YOLOv8_customize/blob/master/README.md")
docs = loader.load()
print(docs[0].page_content)

loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
print(docs[0].page_content)
