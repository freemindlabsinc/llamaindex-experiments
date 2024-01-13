# Debug stuff
import os
import readline
print("Current Working Directory:", os.getcwd())
#env_var = os.getenv('OPENAI_API_KEY')
#print(env_var)

# Sets llama-index
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("python/data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print("---------------------")
print(response)
print("---------------------")

