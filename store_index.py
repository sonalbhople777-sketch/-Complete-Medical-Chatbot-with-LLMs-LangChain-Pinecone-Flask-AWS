import os
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split
from src.helper import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')


os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY


extracted_data=load_pdf_files("data")
minimal_docs=filter_to_minimal_docs(extracted_data)
texts_chunk=text_split(minimal_docs)
embedding = HuggingFaceEmbeddings()

pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)


index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
    
index=pc.Index(index_name)

docsearch=PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)
