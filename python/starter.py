# Import necessary libraries
import logging
import sys
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os

# Set up logging to output to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Uncomment the following lines if you want to print an environment variable
#env_var = os.getenv('OPENAI_API_KEY')
#print(env_var)

# Load documents from the "./data" directory
documents = SimpleDirectoryReader("./data").load_data()

# Create an index from the documents, showing progress as it's created
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Query the engine with a question and store the response
response = query_engine.query("What did the author do growing up?")

# Print the response
print("---------------------")
print(response)
print("---------------------")