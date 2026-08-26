from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents
def filter_to_minimal_docs(docs:List[Document])-> List[Document]:
    """
    given a list of document objects, return a new list of document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] =[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs

## split the document into chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
       
    )
    texts_chunk=text_splitter.split_documents(minimal_docs)
    return texts_chunk

from langchain_huggingface import HuggingFaceEmbeddings

def download_embeddings():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name,
        
    )
    return embeddings