import asyncio
import logging
from elasticsearch import AsyncElasticsearch
from llama_index.vector_stores import ElasticsearchStore
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding    
from llama_index import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import SimpleDirectoryReader

import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
bulk_data = True

# ----------------------------------

async def connect_to_elasticsearch():
    # Instantiate the Elasticsearch client right away to check we can connect

    es_client = AsyncElasticsearch(
        [os.getenv("ES_URL")],
        ssl_assert_fingerprint=os.getenv("ES_CERTIFICATE_FINGERPRINT"),
        basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD"))
    )
    
    await es_client.info() # this connects to the cluster and gets its version
    if bulk_data:        
        await es_client.indices.delete(index=os.getenv("ES_DEFAULT_INDEX"), ignore=[400, 404])

    return es_client



def load_data(es_client):
    # Creates a reader for the /data folder        
    if bulk_data:
        documents = SimpleDirectoryReader("./data").load_data(show_progress=True)

    # Creates the ES vector store
    
    ES_DEFAULT_INDEX = os.getenv("ES_DEFAULT_INDEX")

    es_vector_store = ElasticsearchStore(
        index_name=ES_DEFAULT_INDEX,
        es_client=es_client ,
        
    )

    # Service ctx for debug
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    
        
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    service_context = ServiceContext.from_defaults(
        # callback_manager=callback_manager, 
        llm=llm,
        embed_model=embed_model
    )

    # Creates the index
    import llama_index
    llama_index.set_global_handler("simple")


    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)
    
    if bulk_data:
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            service_context=service_context
        )
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store=es_vector_store, 
            service_context=service_context)        
    
    
    # experiments
    loader = YoutubeTranscriptReader()
    yt_documents = loader.load_data(ytlinks=['https://www.youtube.com/watch?v=i3OYlaoj-BM'])

    index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, service_context=service_context
        )

    return index

async def main():
    es_client = await connect_to_elasticsearch()
    index = load_data(es_client)
    
    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine()
    
    # What is Prince and what can you tell me about Hyphenation?
    while (True):
        question = input("Enter your question: ")
        if question == "":
            question = "what is the address of the bank of yes logic?"
        response = query_engine.query(question)
        
        print("**************************** REFERENCES ****************************")
        print("Refs " + str(response.source_nodes))
        print("**************************** Q&A ****************************")
        print("Q: " + question)
        print("A: " + str(response))        

asyncio.run(main())

