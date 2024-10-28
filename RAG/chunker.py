from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size =26
chunk_overlap = 4
c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = '\n\n '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space. \
and words are separated by space. \n\n  \
my name is kasra sehat. I was born in ardabil which has  amountainary view. there is the sunset is very beautifun and it gives me great feelings.\
from my childhood i loved soccer and i wanted to be one the best soccer player. i worked hard to migrate to tehran after my school.
"""
c_splitter.split_text(some_text)