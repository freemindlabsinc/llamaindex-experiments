import os
from elasticsearch import AsyncElasticsearch
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
    
async def create_elastic_client(deleteIndex: bool = False) -> AsyncElasticsearch:
    es_client = AsyncElasticsearch(
        [os.getenv("ES_URL")],
        ssl_assert_fingerprint = os.getenv("ES_CERTIFICATE_FINGERPRINT"),
        basic_auth = (os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD"))
    )
    
    await es_client.info() # Connects to the cluster and gets its version
    if deleteIndex: 
        await es_client.indices.delete(index=os.getenv("ES_DEFAULT_INDEX"), ignore=[400, 404])        

    return es_client
    
def create_service_context() -> ServiceContext:
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)    
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model) 

    return service_context

def create_vector_store(es_client: AsyncElasticsearch) -> ElasticsearchStore:
    idx_name = os.getenv("ES_DEFAULT_INDEX")
    
    es_vector_store = ElasticsearchStore(
        index_name = idx_name,
        es_client = es_client,
    )

    return es_vector_store

async def bulk_from_local_folder(deleteIndex: bool = True) -> VectorStoreIndex:
    es_client = await create_elastic_client(deleteIndex)
    es_vector_store = create_vector_store(es_client)
    service_context = create_service_context()
    storage_context = StorageContext.from_defaults(vector_store = es_vector_store)

    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)    
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        service_context=service_context,        
        show_progress=True
    )
     
    return index

async def load_existing_data() -> VectorStoreIndex:
    es_client = await create_elastic_client(deleteIndex = False)    
    es_vector_store = create_vector_store(es_client)
    service_context = create_service_context()    

    index = VectorStoreIndex.from_vector_store(
           vector_store=es_vector_store, 
           service_context=service_context,
           show_progress=True)    
    
    return index

def load_data(loader: BaseReader, folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        doc.id_ = doc.metadata["file_name"]
    return docs

async def load_from_googledrive(deleteIndex: bool = False) -> VectorStoreIndex:
    es_client = await create_elastic_client(deleteIndex)    
    es_vector_store = create_vector_store(es_client)
    service_context = create_service_context()
    storage_context = StorageContext.from_defaults(vector_store = es_vector_store)

    GoogleDriveReader = download_loader("GoogleDriveReader")
    loader = GoogleDriveReader()
    #### Using folder id
    documents = loader.load_data(folder_id="1ZBu81ymVvUWn0XgSzYAeBGQ-jyXxOf9E")
    #documents = loader.load_data(file_ids= ["1jfhSUgE0wIoceFzoVz2sDHjUVnh7cTYf"])

    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        service_context=service_context,
        show_progress=True,
        
    )

    return index  