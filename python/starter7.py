print("Initializing")

import loggingUtils as logging
import llamaUtils2 as llama

delete_index = True

async def main():    
    #index = await llama.bulk_from_local_folder(start_fresh);
    index = await llama.load_from_googledrive2(delete_index)
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
        idx = 1
        for src in response.source_nodes:                      
            file_name = src.metadata["file_name"]
            page = src.metadata["page_label"]
            mime = src.metadata["mime type"]
            created = src.metadata["created at"]
            modified = src.metadata["modified at"]
            
            print(f"[Ref #{idx}] Page {page} of {file_name}. Last modified {modified}, MIME {mime}.")
            idx += 1
            
        print("**************************** Q&A ****************************")
        print("Q: " + question)
        print("A: " + str(response))        

import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

