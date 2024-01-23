import loggingUtils as logging
import llamaUtils as llama
import asyncio

delete_index = True

async def main():    
    #index = await llama.bulk_from_local_folder(deleteIndex=delete_index);
    index = await llama.load_from_googledrive(delete_index)
    #index = await llama.load_existing_data(delete_index);

    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine()
    
    # What is Prince and what can you tell me about Hyphenation?
    while (True):
        question = input("Enter your question: ")
        if question == "":
            question = "what is the program prince?"
        response = query_engine.query(question)
        
        print("**************************** REFERENCES ****************************")
        print("Refs " + str(response.source_nodes))
        print("**************************** Q&A ****************************")
        print("Q: " + question)
        print("A: " + str(response))        


asyncio.run(main())