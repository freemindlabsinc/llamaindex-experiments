# Import necessary modules
import logging
import sys
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Define the directory where the index will be stored
PERSIST_DIR = "./.storage"

# Check if the storage directory already exists
if not os.path.exists(PERSIST_DIR):
    # If it doesn't exist, load the documents and create the index
    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)

    # Persist the index for later use
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # If the storage directory exists, load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, show_progress=True)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Query the index and print the response
response = query_engine.query("What did the author do growing up?")
print(response)