import loggingUtils as logging
import llamautils as llama

logging.setup_logging()

start_fresh = False


async def main():    
    if (start_fresh):
        index = await llama.bulk_from_local_folder();
    else:
        index = await llama.load_existing_data();

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

import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

