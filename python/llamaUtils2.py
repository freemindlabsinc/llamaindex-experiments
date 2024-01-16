from json.encoder import ESCAPE_ASCII
import os
from elasticsearch import AsyncElasticsearch
from elasticsearch._async.client import nodes
from llama_index import SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.indices import vector_store
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.readers.base import BaseReader    
from llama_index.vector_stores import ElasticsearchStore
from llama_index.storage.storage_context import StorageContext
from llama_index import VectorStoreIndex
from llama_index import download_loader
# --- new
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.ingestion.cache import RedisCache
from llama_index.storage.docstore import BaseDocumentStore, DocumentStore, RedisDocumentStore, SimpleDocumentStore, redis_docstore
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores.types import BasePydanticVectorStore
    
def create_service_context() -> ServiceContext:
    llm = OpenAI(model= os.getenv("OPENAI_CHAT_MODEL"), temperature= os.getenv("OPENAI_MODEL_TEMPERATURE"))    
    embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model) 

    return service_context

def create_vector_store() -> BasePydanticVectorStore:
    es_client = AsyncElasticsearch(
        [os.getenv("ES_URL")],
        ssl_assert_fingerprint = os.getenv("ES_CERTIFICATE_FINGERPRINT"),
        basic_auth = (os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD"))
    )
    idx_name = os.getenv("ES_DEFAULT_INDEX")
    
    es_vector_store = ElasticsearchStore(
        index_name = idx_name,
        es_client = es_client,
    )        

    return es_vector_store

def create_document_store() -> BaseDocumentStore:
    redis_doc_store = RedisDocumentStore.from_host_and_port(
        host=os.getenv("REDIS_SERVER"), 
        port=os.getenv("REDIS_PORT"), 
        namespace=os.getenv("REDIS_DOCUMENT_STORE_NAMESPACE")
    )
    return redis_doc_store;

def create_ingestion_cache() -> IngestionCache:
    cache = IngestionCache(
        cache=RedisCache.from_host_and_port(
            host=os.getenv("REDIS_SERVER"), 
            port=os.getenv("REDIS_PORT"), 
        ),
        collection=os.getenv("REDIS_CACHE_COLLECTION"),
    )

    return cache;

def custom_load_data(loader: BaseReader, folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    count = 1
    for doc in docs:
        #doc.id_ = doc.metadata["file_name"]
        count += 1
        
    print(f"Loaded {count} nodes.")
    
    return docs

def update_llama(pipeline: IngestionPipeline):
    GoogleDriveReader = download_loader("GoogleDriveReader")
    loader = GoogleDriveReader()
    documents = custom_load_data(loader=loader, folder_id=os.getenv("GOOGLE_FOLDER"))
    pipeline.run(documents=documents, show_progress=True)

    return

def load_from_googledrive2(deleteIndex: bool = False) -> (VectorStoreIndex, IngestionPipeline):    
    vector_store = create_vector_store()
    service_context = create_service_context()
    doc_store = create_document_store()    
    cache = create_ingestion_cache();
    embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))

    pipeline = IngestionPipeline(        
        transformations=[
            SentenceSplitter(),
            embed_model,
        ],
        docstore=doc_store,
        vector_store=vector_store,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,        
    )
    
    update_llama(pipeline);

    storage_context = StorageContext.from_defaults(
        vector_store = vector_store
    )    
    
    index = VectorStoreIndex.from_vector_store(
        storage_context=storage_context, 
        service_context=service_context,
        vector_store=vector_store,
        show_progress=True,        
    )    
    
    return index, pipeline