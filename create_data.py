from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
from dotenv import load_dotenv

load_dotenv()


def load_documents(path):
    loader = DirectoryLoader(path=path, glob="*.md")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    document = chunks[10]

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists("chroma"):
        shutil.rmtree("chroma")

    # Create a new DB from the documents.
    print(len(chunks))
    if len(chunks) > 0:
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory="chroma"
        )
    else:
        db = Chroma(persist_directory="chroma", embedding_function=OpenAIEmbeddings())
    db.persist()
    print("Database created")
    return db
