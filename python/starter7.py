print("Initializing")

import loggingUtils as logging
import llamaUtils2 as llama
from colorama import Fore

delete_index = True

async def main():    
    #index = await llama.bulk_from_local_folder(start_fresh);
    index = await llama.load_from_googledrive2(delete_index)
    #index = await llama.load_existing_data(delete_index);

    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine()
    
    # What is Prince and what can you tell me about Hyphenation?
    while (True):
        question = input(Fore.WHITE + "Enter your question: ")
        if question == "":
            question = "what is the program prince?"
        
        #question += " [RESPOND IN ITALIAN]";

        response = query_engine.query(question)
        
        print("**************************** REFERENCES ****************************")
        idx = 1
        for src in response.source_nodes:                      
            file_name = src.metadata.get("file name", "UNKNOWN")
            page = src.metadata.get("page_label", "None")
            mime = src.metadata.get("mime type", "NO-MIME")
            created = src.metadata.get("created at", "Sometime")
            modified = src.metadata.get("modified at", "Sometime after")
            
            print(Fore.LIGHTBLACK_EX + f"[Ref #{idx}] Page {page} of {file_name}. Last modified {modified}, MIME {mime}.")
            idx += 1
            
        print(Fore.GREEN + "**************************** Q&A ****************************")
        print("Q: " + question)
        print("A: " + str(response))        

import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

