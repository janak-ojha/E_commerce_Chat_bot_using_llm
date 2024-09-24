from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # Hypothetical import; adjust based on actual library
from dotenv import load_dotenv
import os
import pandas as pd
from ecommbot.data_converter import dataconveter

load_dotenv()

# Remove OpenAI API key and replace it with the Hugging Face API key if needed
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Update environment variable as necessary
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

# Initialize the embeddings using Hugging Face
embedding = HuggingFaceEmbeddings(api_key=HUGGINGFACE_API_KEY)  # Update the initialization as needed

def ingestdata(status):
    vstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )
    
    storage = status
    
    if storage is None:
        docs = dataconveter()
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    return vstore, inserted_ids

if __name__ == '__main__':
    vstore, inserted_ids = ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
