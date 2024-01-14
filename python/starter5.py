from azure.storage.queue import QueueServiceClient
import json
import os

from dotenv import load_dotenv
load_dotenv()

# Azure Queue connection string and queue name
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
notifications_queue_name = os.getenv("NOTIFICATIONS_QUEUE_NAME")

# Initialize the QueueServiceClient
queue_service = QueueServiceClient.from_connection_string(connect_str)
queue_client = queue_service.get_queue_client(notifications_queue_name)

# Continuously read and process messages
while True:
    messages = queue_client.receive_messages()

    for msg in messages:
        try:
            # Assuming the message is in JSON format
            message_content = json.loads(msg.content)
            print("Received message:", message_content)

            # Process the message (your logic here)

            # Delete the message after processing
            queue_client.delete_message(msg)
        except json.JSONDecodeError:
            print("Error: Message is not in valid JSON format")